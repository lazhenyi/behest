//! RunService gRPC implementation.

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::grpc::pb::{
    AgentEvent as PbAgentEvent, CancelRunRequest, CancelRunResponse, CreateRunRequest,
    CreateRunResponse, GetRunOutputRequest, GetRunOutputResponse, GetRunRequest, GetRunResponse,
    ListRunsRequest, ListRunsResponse, RunOutput as PbRunOutput, RunRecord as PbRunRecord,
    RunStatus as PbRunStatus, WatchRunEventsRequest, run_service_server::RunService,
};

use crate::grpc::event::to_proto;
use crate::provider::{ModelName, ProviderId};
use crate::runtime::RunId;
use crate::runtime::RunRequest;
use crate::runtime::run::{RunRecord, RunStatus};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};

/// In-memory registry of active run tasks for cancellation.
#[derive(Default)]
pub struct RunTaskRegistry {
    handles: RwLock<HashMap<String, tokio::task::JoinHandle<()>>>,
}

impl RunTaskRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a run task.
    pub async fn register(&self, run_id: String, handle: tokio::task::JoinHandle<()>) {
        self.handles.write().await.insert(run_id, handle);
    }

    /// Aborts and removes a run task.
    pub async fn cancel(&self, run_id: &str) -> bool {
        let mut handles = self.handles.write().await;
        if let Some(handle) = handles.remove(run_id) {
            handle.abort();
            true
        } else {
            false
        }
    }
}

/// gRPC run service.
pub struct GrpcRunService {
    state: Arc<super::state::GrpcState>,
    tasks: Arc<RunTaskRegistry>,
}

impl GrpcRunService {
    /// Creates a new run service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>, tasks: Arc<RunTaskRegistry>) -> Self {
        Self { state, tasks }
    }
}

#[tonic::async_trait]
impl RunService for GrpcRunService {
    async fn create_run(
        &self,
        request: Request<CreateRunRequest>,
    ) -> Result<Response<CreateRunResponse>, Status> {
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
        let tasks = Arc::clone(&self.tasks);

        let handle = tokio::spawn(async move {
            if let Err(e) = runtime.run(run_request).await {
                tracing::error!(error = %e, "asynchronous run failed");
            }
        });

        let run_id = Uuid::new_v4().to_string();
        tasks.register(run_id.clone(), handle).await;

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
        tokio::spawn(async move {
            if let Err(e) = runtime.run(run_request).await {
                tracing::error!(error = %e, "streaming run failed");
            }
        });

        // Forward broadcast events for this run, filtered by run_id obtained
        // from the first RunStarted event.
        tokio::spawn(async move {
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

        let runs = self
            .state
            .runtime
            .store()
            .runs()
            .list_runs_filtered(session_id, status, limit as usize, offset as usize)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

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
            .store()
            .runs()
            .get_run(run_id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
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

        let Some(run) = self
            .state
            .runtime
            .store()
            .runs()
            .get_run(run_id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        else {
            return Err(Status::not_found("run not found"));
        };

        if !run.status.is_terminal() {
            return Err(Status::failed_precondition("run has not completed yet"));
        }

        let finish_reason = crate::grpc::pb::FinishReason::Unspecified as i32;

        Ok(Response::new(GetRunOutputResponse {
            output: Some(PbRunOutput {
                run_id: run_id.to_string(),
                session_id: run.session_id.to_string(),
                iterations: 0,
                finish_reason,
                total_usage: None,
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
            .store()
            .runs()
            .get_run(run_id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        else {
            return Err(Status::not_found("run not found"));
        };

        if run.status.is_terminal() {
            return Err(Status::failed_precondition(
                "run is already in a terminal state",
            ));
        }

        self.tasks.cancel(&run_id.to_string()).await;

        self.state
            .runtime
            .store()
            .runs()
            .update_run_status(run_id, RunStatus::Cancelled)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

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

        // Replay persisted events after last_event_id.
        let events = self
            .state
            .runtime
            .store()
            .runs()
            .list_events(run_id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut max_seq = last_event_id;
        for record in &events {
            if record.sequence > last_event_id {
                let pb = to_proto(&record.event, record.sequence, &run_id.to_string());
                max_seq = record.sequence;
                if tx.send(Ok(pb)).await.is_err() {
                    return Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
                        rx,
                    )));
                }
            }
        }

        // Check if run is already terminal.
        if let Ok(Some(run)) = self.state.runtime.store().runs().get_run(run_id).await {
            if run.status.is_terminal() {
                drop(tx);
                return Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
                    rx,
                )));
            }
        }

        // Subscribe to live events.
        let mut broadcast_rx = self.state.runtime.subscribe();
        let rid = run_id;
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
        provider: Some(crate::grpc::pb::ProviderId {
            value: r.provider.to_string(),
        }),
        model: Some(crate::grpc::pb::ModelName {
            value: r.model.as_str().to_owned(),
        }),
        metadata: r.metadata.to_string(),
        created_at: Some(crate::grpc::pb::Timestamp {
            value: r.created_at.to_rfc3339(),
        }),
        updated_at: Some(crate::grpc::pb::Timestamp {
            value: r.updated_at.to_rfc3339(),
        }),
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
