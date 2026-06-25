//! gRPC ChatService implementation — raw provider passthrough streaming.

use std::sync::Arc;

use futures_util::StreamExt as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::grpc::pb::{
    ChatStreamEvent as PbChatStreamEvent, ChatStreamFinished, ChatTextDelta,
    ChatToolCallArgumentsDelta, ChatToolCallCompleted, ChatToolCallStarted,
    chat_stream_event::Event as ChatEventKind,
};
use crate::grpc::state::GrpcState;
use crate::provider::{ChatRequest, ModelName, ProviderId};

use super::pb::chat_service_server::ChatService;
use super::pb::{ChatStreamStarted, RawChatRequest};

/// gRPC ChatService: raw provider streaming without runtime orchestration.
pub struct GrpcChatService {
    state: Arc<GrpcState>,
}

impl GrpcChatService {
    /// Creates a new ChatService backed by the given runtime state.
    #[must_use]
    pub fn new(state: Arc<GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ChatService for GrpcChatService {
    type RawChatStreamStream = ReceiverStream<Result<PbChatStreamEvent, Status>>;

    async fn raw_chat_stream(
        &self,
        request: Request<RawChatRequest>,
    ) -> Result<Response<Self::RawChatStreamStream>, Status> {
        let req = request.into_inner();

        let provider_id = req
            .provider
            .as_ref()
            .and_then(|p| {
                if p.value.is_empty() {
                    None
                } else {
                    Some(&p.value)
                }
            })
            .ok_or_else(|| Status::invalid_argument("provider is required"))?;
        let provider_id = ProviderId::new(provider_id.clone());

        let model = req
            .model
            .as_ref()
            .and_then(|m| {
                if m.value.is_empty() {
                    None
                } else {
                    Some(&m.value)
                }
            })
            .ok_or_else(|| Status::invalid_argument("model is required"))?;
        let model = ModelName::new(model.clone());

        if req.input.is_empty() {
            return Err(Status::invalid_argument("input must not be empty"));
        }

        let chat_request = ChatRequest::new(model).with_user_text(&req.input);

        let mut stream = self
            .state
            .runtime
            .providers()
            .stream(&provider_id, chat_request)
            .await
            .map_err(|e| super::provider_error_to_status(&e))?;

        let (tx, rx) = mpsc::channel(256);

        tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                let event = match result {
                    Ok(evt) => evt,
                    Err(e) => {
                        let _ = tx.send(Err(super::provider_error_to_status(&e))).await;
                        return;
                    }
                };

                let pb = chat_stream_event_to_proto(event);
                if tx.send(Ok(pb)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn chat_stream_event_to_proto(event: crate::provider::ChatStreamEvent) -> PbChatStreamEvent {
    use crate::provider::ChatStreamEvent;

    let kind = match event {
        ChatStreamEvent::Started { provider, model } => ChatEventKind::Started(ChatStreamStarted {
            provider: provider.to_string(),
            model: model.to_string(),
        }),
        ChatStreamEvent::TextDelta { delta } => ChatEventKind::TextDelta(ChatTextDelta { delta }),
        ChatStreamEvent::ToolCallStarted { id, name } => {
            ChatEventKind::ToolCallStarted(ChatToolCallStarted { id, name })
        }
        ChatStreamEvent::ToolCallArgumentsDelta { id, delta } => {
            ChatEventKind::ToolCallArgumentsDelta(ChatToolCallArgumentsDelta { id, delta })
        }
        ChatStreamEvent::ToolCallCompleted { call } => {
            ChatEventKind::ToolCallCompleted(ChatToolCallCompleted {
                call: Some(super::pb::ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments.to_string(),
                }),
            })
        }
        ChatStreamEvent::Finished {
            finish_reason,
            usage,
        } => ChatEventKind::Finished(ChatStreamFinished {
            finish_reason: super::event::finish_reason_to_proto(&finish_reason),
            usage: usage.map(|u| super::pb::TokenUsage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
            }),
        }),
    };

    PbChatStreamEvent { event: Some(kind) }
}
