//! RunService gRPC implementation.
//!
//! Provides RPCs for creating runs (with optional streaming events),
//! listing/filtering runs, retrieving run output, cancelling runs,
//! and watching live run events via server-sent streaming.

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::transport::grpc::pb::{
    AgentEvent as PbAgentEvent, CancelRunRequest, CancelRunResponse, CreateRunRequest,
    CreateRunResponse, GetRunOutputRequest, GetRunOutputResponse, GetRunRequest, GetRunResponse,
    ListRunsRequest, ListRunsResponse, RunOutput as PbRunOutput, RunRecord as PbRunRecord,
    RunStatus as PbRunStatus, TokenUsage as PbTokenUsage, WatchRunEventsRequest,
    run_service_server::RunService,
};

use crate::provider::{ModelName, ProviderId};
use crate::runtime::RunId;
use crate::runtime::RunRequest;
use crate::runtime::run::{RunRecord, RunStatus};
use crate::transport::grpc::event::to_proto;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};

/// Parses the `grpc-timeout` header from a tonic request into a [`Duration`].
fn parse_grpc_timeout<T>(request: &Request<T>) -> Option<Duration> {
    let val = request.metadata().get("grpc-timeout")?.to_str().ok()?;
    let (num_str, unit) = val.split_at(val.len() - 1);
    let num: u64 = num_str.parse().ok()?;
    match unit {
        "H" => Some(Duration::from_secs(num * 3600)),
        "M" => Some(Duration::from_secs(num * 60)),
        "S" => Some(Duration::from_secs(num)),
        "m" => Some(Duration::from_millis(num)),
        "u" => Some(Duration::from_micros(num)),
        "n" => Some(Duration::from_nanos(num)),
        _ => None,
    }
}

/// In-memory registry of active run tasks for cancellation and idempotency.
///
/// Tracks spawned tokio tasks by run ID for coordinated cancellation
/// and maintains a client-request-ID to run-ID mapping for at-most-once
/// run creation semantics.
#[derive(Default)]
pub struct RunTaskRegistry {
    handles: RwLock<HashMap<String, tokio::task::JoinHandle<()>>>,
    /// Maps client_request_id to run_id for idempotency.
    idempotency_cache: RwLock<HashMap<String, String>>,
}

impl RunTaskRegistry {
    /// Creates a new empty run task registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a run task under the given run ID.
    ///
    /// The associated `JoinHandle` is stored for later cancellation.
    pub async fn register(&self, run_id: String, handle: tokio::task::JoinHandle<()>) {
        self.handles.write().await.insert(run_id, handle);
    }

    /// Aborts and removes a run task by run ID.
    ///
    /// Returns `true` if the task was found and aborted.
    pub async fn cancel(&self, run_id: &str) -> bool {
        let mut handles = self.handles.write().await;
        if let Some(handle) = handles.remove(run_id) {
            handle.abort();
            true
        } else {
            false
        }
    }

    /// Returns the number of currently registered (active) run tasks.
    pub async fn active_count(&self) -> usize {
        self.handles.read().await.len()
    }

    /// Returns the previously-assigned run ID for a client request ID, if any.
    ///
    /// Used for idempotent run creation: duplicate requests with the same
    /// client-generated identifier return the existing run instead of
    /// creating a new one.
    pub async fn get_idempotent_run_id(&self, client_request_id: &str) -> Option<String> {
        self.idempotency_cache
            .read()
            .await
            .get(client_request_id)
            .cloned()
    }

    /// Records a mapping from client request ID to run ID for idempotency.
    pub async fn record_idempotent_run(&self, client_request_id: String, run_id: String) {
        self.idempotency_cache
            .write()
            .await
            .insert(client_request_id, run_id);
    }
}

/// gRPC run service for agent execution lifecycle management.
///
/// Supports creating runs (sync and streaming), listing/filtering
/// runs by session and status, retrieving completed output, cancelling
/// active runs, and watching run events in real time.
pub struct GrpcRunService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcRunService {
    /// Creates a new run service backed by the given shared state.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl RunService for GrpcRunService {
    async fn create_run(
        &self,
        request: Request<CreateRunRequest>,
    ) -> Result<Response<CreateRunResponse>, Status> {
        let deadline = parse_grpc_timeout(&request);
        let req = request.into_inner();
        let provider_id = req
            .provider
            .as_ref()
            .map(|p| ProviderId::new(&p.value))
            .ok_or_else(|| Status::invalid_argument("provider is required"))?;
        let model = req
            .model
            .as_ref()
            .map(|m| ModelName::new(&m.value))
            .ok_or_else(|| Status::invalid_argument("model is required"))?;

        if req.input.is_empty() {
            return Err(Status::invalid_argument("input must not be empty"));
        }

        if let Some(max) = self.state.max_concurrent_runs {
            let active = self.state.run_tasks.active_count().await;
            if active >= max {
                return Err(Status::resource_exhausted(format!(
                    "concurrent run limit reached ({max})"
                )));
            }
        }

        let session_id = if req.session_id.is_empty() {
            None
        } else {
            Some(
                Uuid::parse_str(&req.session_id)
                    .map_err(|_| Status::invalid_argument("invalid session id"))?,
            )
        };

        let run_id = RunId::new();
        let mut run_request = RunRequest::new(provider_id, model, &req.input).with_run_id(run_id);
        if let Some(sid) = session_id {
            run_request = run_request.with_session_id(sid);
        }
        if !req.client_request_id.is_empty() {
            if let Some(existing_run_id) = self
                .state
                .run_tasks
                .get_idempotent_run_id(&req.client_request_id)
                .await
            {
                return Ok(Response::new(CreateRunResponse {
                    run_id: existing_run_id,
                    session_id: session_id.map_or_else(String::new, |s| s.to_string()),
                }));
            }
            run_request = run_request.with_client_request_id(req.client_request_id.clone());
        }

        let runtime = Arc::clone(&self.state.runtime);
        let tasks = Arc::clone(&self.state.run_tasks);

        let handle = tokio::spawn(async move {
            let run_future = async {
                if let Err(e) = runtime.run(run_request).await {
                    tracing::error!(error = %e, "asynchronous run failed");
                }
            };
            if let Some(d) = deadline {
                if tokio::time::timeout(d, run_future).await.is_err() {
                    tracing::warn!("run cancelled: gRPC deadline exceeded");
                }
            } else {
                run_future.await;
            }
        });

        let run_id = run_id.to_string();
        tasks.register(run_id.clone(), handle).await;

        if !req.client_request_id.is_empty() {
            tasks
                .record_idempotent_run(req.client_request_id.clone(), run_id.clone())
                .await;
        }

        Ok(Response::new(CreateRunResponse {
            run_id,
            session_id: session_id.map_or_else(String::new, |s| s.to_string()),
        }))
    }

    type CreateRunStreamStream =
        tokio_stream::wrappers::ReceiverStream<Result<PbAgentEvent, Status>>;

    async fn create_run_stream(
        &self,
        request: Request<CreateRunRequest>,
    ) -> Result<Response<Self::CreateRunStreamStream>, Status> {
        let deadline = parse_grpc_timeout(&request);
        let req = request.into_inner();
        let provider_id = req
            .provider
            .as_ref()
            .map(|p| ProviderId::new(&p.value))
            .ok_or_else(|| Status::invalid_argument("provider is required"))?;
        let model = req
            .model
            .as_ref()
            .map(|m| ModelName::new(&m.value))
            .ok_or_else(|| Status::invalid_argument("model is required"))?;

        if req.input.is_empty() {
            return Err(Status::invalid_argument("input must not be empty"));
        }
        let session_id = if req.session_id.is_empty() {
            None
        } else {
            Some(
                Uuid::parse_str(&req.session_id)
                    .map_err(|_| Status::invalid_argument("invalid session id"))?,
            )
        };

        let run_request = RunRequest::new(provider_id, model, &req.input);
        let run_request = if let Some(sid) = session_id {
            run_request.with_session_id(sid)
        } else {
            run_request
        };

        let runtime = Arc::clone(&self.state.runtime);
        let mut broadcast_rx = runtime.subscribe();
        let (tx, rx) = mpsc::channel(256);

        // Spawn the run in background.
        let run_deadline = deadline;
        tokio::spawn(async move {
            let run_future = async {
                if let Err(e) = runtime.run(run_request).await {
                    tracing::error!(error = %e, "streaming run failed");
                }
            };
            if let Some(d) = run_deadline {
                if tokio::time::timeout(d, run_future).await.is_err() {
                    tracing::warn!("streaming run cancelled: gRPC deadline exceeded");
                }
            } else {
                run_future.await;
            }
        });

        // Forward broadcast events for this run, filtered by run_id obtained
        // from the first RunStarted event.
        let stream_deadline = deadline;
        tokio::spawn(async move {
            let forward = async {
                let mut sequence: u64 = 0;
                let mut run_id: Option<RunId> = None;

                loop {
                    match broadcast_rx.recv().await {
                        Ok(event) => {
                            if run_id.is_none() {
                                run_id = Some(event.run_id());
                            }
                            let Some(rid) = run_id else {
                                continue;
                            };
                            if event.run_id() != rid {
                                continue;
                            }

                            let is_terminal = event.is_terminal();
                            let pb = to_proto(&event, sequence, &rid.to_string());
                            sequence += 1;

                            if tx.send(Ok(pb)).await.is_err() {
                                return;
                            }
                            if is_terminal {
                                return;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(skipped = n, "broadcast receiver lagged");
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            return;
                        }
                    }
                }
            };
            if let Some(d) = stream_deadline {
                if tokio::time::timeout(d, forward).await.is_err() {
                    tracing::warn!("event stream terminated: gRPC deadline exceeded");
                }
            } else {
                forward.await;
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }

    async fn list_runs(
        &self,
        request: Request<ListRunsRequest>,
    ) -> Result<Response<ListRunsResponse>, Status> {
        let req = request.into_inner();
        let session_id = if req.session_id.is_empty() {
            None
        } else {
            Some(
                Uuid::parse_str(&req.session_id)
                    .map_err(|_| Status::invalid_argument("invalid session id"))?,
            )
        };

        let status = run_status_from_pb(req.status());
        let limit = req.pagination.as_ref().map_or(100, |p| p.limit);
        let offset = req.pagination.as_ref().map_or(0, |p| p.offset);

        if limit > 100 {
            return Err(Status::invalid_argument(
                "pagination limit exceeds maximum of 100",
            ));
        }

        let runs = self
            .state
            .runtime
            .runs()
            .list_runs_filtered(session_id, status, limit as usize, offset as usize)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?;

        let pb_runs: Vec<PbRunRecord> = runs.iter().map(run_record_to_proto).collect();

        Ok(Response::new(ListRunsResponse { runs: pb_runs }))
    }

    async fn get_run(
        &self,
        request: Request<GetRunRequest>,
    ) -> Result<Response<GetRunResponse>, Status> {
        let req = request.into_inner();
        let run_id =
            parse_run_id(&req.run_id).map_err(|_| Status::invalid_argument("invalid run id"))?;

        let Some(run) = self
            .state
            .runtime
            .runs()
            .get_run(run_id)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?
        else {
            return Err(Status::not_found("run not found"));
        };

        Ok(Response::new(GetRunResponse {
            run: Some(run_record_to_proto(&run)),
        }))
    }

    async fn get_run_output(
        &self,
        request: Request<GetRunOutputRequest>,
    ) -> Result<Response<GetRunOutputResponse>, Status> {
        let req = request.into_inner();
        let run_id =
            parse_run_id(&req.run_id).map_err(|_| Status::invalid_argument("invalid run id"))?;

        let Some(state) = self
            .state
            .runtime
            .runs()
            .get_run_state(run_id)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?
        else {
            return Err(Status::not_found("run not found"));
        };

        if !state.status.is_terminal() {
            return Err(Status::failed_precondition("run has not completed yet"));
        }

        let finish_reason = state.last_finish.as_ref().map_or(
            crate::transport::grpc::pb::FinishReason::Unspecified as i32,
            super::event::finish_reason_to_proto,
        );

        Ok(Response::new(GetRunOutputResponse {
            output: Some(PbRunOutput {
                run_id: run_id.to_string(),
                session_id: state.session_id.to_string(),
                iterations: u32::try_from(state.iteration).unwrap_or(u32::MAX),
                finish_reason,
                total_usage: Some(PbTokenUsage {
                    input_tokens: state.total_usage.input_tokens,
                    output_tokens: state.total_usage.output_tokens,
                    total_tokens: state.total_usage.total_tokens,
                }),
                messages: Vec::new(),
            }),
        }))
    }

    async fn cancel_run(
        &self,
        request: Request<CancelRunRequest>,
    ) -> Result<Response<CancelRunResponse>, Status> {
        let req = request.into_inner();
        let run_id =
            parse_run_id(&req.run_id).map_err(|_| Status::invalid_argument("invalid run id"))?;

        let Some(run) = self
            .state
            .runtime
            .runs()
            .get_run(run_id)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?
        else {
            return Err(Status::not_found("run not found"));
        };

        if run.status.is_terminal() {
            return Err(Status::failed_precondition(
                "run is already in a terminal state",
            ));
        }

        self.state.run_tasks.cancel(&run_id.to_string()).await;

        self.state
            .runtime
            .runs()
            .update_run_status(run_id, RunStatus::Cancelled)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?;

        Ok(Response::new(CancelRunResponse {}))
    }

    type WatchRunEventsStream =
        tokio_stream::wrappers::ReceiverStream<Result<PbAgentEvent, Status>>;

    async fn watch_run_events(
        &self,
        request: Request<WatchRunEventsRequest>,
    ) -> Result<Response<Self::WatchRunEventsStream>, Status> {
        let req = request.into_inner();
        let run_id =
            parse_run_id(&req.run_id).map_err(|_| Status::invalid_argument("invalid run id"))?;
        let last_event_id = req.last_event_id;

        let (tx, rx) = mpsc::channel(256);

        // Subscribe to live events BEFORE replay to avoid losing events
        // that arrive between replay completion and subscription.
        let mut broadcast_rx = self.state.runtime.subscribe();
        let rid = run_id;

        // Replay persisted events after last_event_id.
        let events = self
            .state
            .runtime
            .runs()
            .list_events(run_id)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?;

        let mut max_seq = last_event_id;
        for record in &events {
            if record.sequence > last_event_id {
                let pb = to_proto(&record.event, record.sequence, &rid.to_string());
                max_seq = record.sequence;
                if tx.send(Ok(pb)).await.is_err() {
                    return Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
                        rx,
                    )));
                }
            }
        }

        // Check if run is already terminal.
        if let Ok(Some(run)) = self.state.runtime.runs().get_run(run_id).await {
            if run.status.is_terminal() {
                drop(tx);
                return Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
                    rx,
                )));
            }
        }

        tokio::spawn(async move {
            loop {
                match broadcast_rx.recv().await {
                    Ok(event) => {
                        if event.run_id() != rid {
                            continue;
                        }
                        max_seq += 1;
                        let is_terminal = event.is_terminal();
                        let pb = to_proto(&event, max_seq, &rid.to_string());
                        if tx.send(Ok(pb)).await.is_err() {
                            return;
                        }
                        if is_terminal {
                            return;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "broadcast receiver lagged");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        return;
                    }
                }
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }
}

fn parse_run_id(s: &str) -> Result<RunId, uuid::Error> {
    let id = Uuid::parse_str(s)?;
    Ok(RunId::from_uuid(id))
}

fn run_status_from_pb(status: PbRunStatus) -> Option<RunStatus> {
    match status {
        PbRunStatus::Pending => Some(RunStatus::Pending),
        PbRunStatus::Completed => Some(RunStatus::Completed),
        PbRunStatus::Failed => Some(RunStatus::Failed),
        PbRunStatus::Cancelled => Some(RunStatus::Cancelled),
        _ => None,
    }
}

fn run_record_to_proto(r: &RunRecord) -> PbRunRecord {
    PbRunRecord {
        id: r.id.to_string(),
        session_id: r.session_id.to_string(),
        status: run_status_to_pb(r.status).into(),
        provider: Some(crate::transport::grpc::pb::ProviderId {
            value: r.provider.to_string(),
        }),
        model: Some(crate::transport::grpc::pb::ModelName {
            value: r.model.as_str().to_owned(),
        }),
        metadata: r.metadata.to_string(),
        created_at: Some(crate::transport::grpc::to_prost_timestamp(r.created_at)),
        updated_at: Some(crate::transport::grpc::to_prost_timestamp(r.updated_at)),
    }
}

fn run_status_to_pb(s: RunStatus) -> PbRunStatus {
    match s {
        RunStatus::Pending => PbRunStatus::Pending,
        RunStatus::SessionLoaded => PbRunStatus::SessionLoaded,
        RunStatus::BuildingContext => PbRunStatus::BuildingContext,
        RunStatus::CallingModel => PbRunStatus::CallingModel,
        RunStatus::WaitingForTools => PbRunStatus::WaitingForTools,
        RunStatus::Persisting => PbRunStatus::Persisting,
        RunStatus::Completed => PbRunStatus::Completed,
        RunStatus::Failed => PbRunStatus::Failed,
        RunStatus::Cancelled => PbRunStatus::Cancelled,
    }
}
