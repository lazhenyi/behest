//! SessionService gRPC implementation.

use tonic::{Request, Response, Status};

use crate::grpc::pb::{
    ContentPart, CreateSessionRequest, CreateSessionResponse, DeleteSessionRequest,
    DeleteSessionResponse, GetSessionRequest, GetSessionResponse, ListMessagesRequest,
    ListMessagesResponse, ListSessionsRequest, ListSessionsResponse, Message, Session, Timestamp,
    UpdateSessionRequest, UpdateSessionResponse, session_service_server::SessionService,
};

use super::pb::ModelName;
use std::sync::Arc;

use crate::provider;

/// gRPC session service.
pub struct GrpcSessionService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcSessionService {
    /// Creates a new session service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl SessionService for GrpcSessionService {
    async fn create_session(
        &self,
        request: Request<CreateSessionRequest>,
    ) -> Result<Response<CreateSessionResponse>, Status> {
        let req = request.into_inner();
        let model = req.model.map_or_else(
            || provider::ModelName::new("default"),
            |m| provider::ModelName::new(m.value),
        );

        let sess = crate::store::Session::new(req.title, model);

        let created = self
            .state
            .runtime
            .store()
            .sessions()
            .create_session(sess)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(CreateSessionResponse {
            session: Some(session_to_proto(&created)),
        }))
    }

    async fn list_sessions(
        &self,
        request: Request<ListSessionsRequest>,
    ) -> Result<Response<ListSessionsResponse>, Status> {
        let req = request.into_inner();
        let pagination = crate::store::Pagination {
            limit: req.pagination.as_ref().map_or(100, |p| p.limit),
            offset: req.pagination.as_ref().map_or(0, |p| p.offset),
        };

        let sessions = self
            .state
            .runtime
            .store()
            .sessions()
            .list_sessions_paginated(pagination, crate::store::SessionFilter::default())
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(ListSessionsResponse {
            sessions: sessions.iter().map(session_to_proto).collect(),
        }))
    }

    async fn get_session(
        &self,
        request: Request<GetSessionRequest>,
    ) -> Result<Response<GetSessionResponse>, Status> {
        let req = request.into_inner();
        let id = uuid::Uuid::parse_str(&req.id)
            .map_err(|_| Status::invalid_argument("invalid session id"))?;

        let Some(session) = self
            .state
            .runtime
            .store()
            .sessions()
            .get_session(&id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        else {
            return Err(Status::not_found("session not found"));
        };

        Ok(Response::new(GetSessionResponse {
            session: Some(session_to_proto(&session)),
        }))
    }

    async fn update_session(
        &self,
        request: Request<UpdateSessionRequest>,
    ) -> Result<Response<UpdateSessionResponse>, Status> {
        let req = request.into_inner();
        let id = uuid::Uuid::parse_str(&req.id)
            .map_err(|_| Status::invalid_argument("invalid session id"))?;

        let title = req.title.as_deref().unwrap_or("");
        let model = req.model.map(|m| provider::ModelName::new(m.value));

        let session = self
            .state
            .runtime
            .store()
            .sessions()
            .update_session(&id, title, model.as_ref())
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(UpdateSessionResponse {
            session: Some(session_to_proto(&session)),
        }))
    }

    async fn delete_session(
        &self,
        request: Request<DeleteSessionRequest>,
    ) -> Result<Response<DeleteSessionResponse>, Status> {
        let req = request.into_inner();
        let id = uuid::Uuid::parse_str(&req.id)
            .map_err(|_| Status::invalid_argument("invalid session id"))?;

        self.state
            .runtime
            .store()
            .sessions()
            .delete_session(&id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(DeleteSessionResponse {}))
    }

    async fn list_messages(
        &self,
        request: Request<ListMessagesRequest>,
    ) -> Result<Response<ListMessagesResponse>, Status> {
        let req = request.into_inner();
        let session_id = uuid::Uuid::parse_str(&req.session_id)
            .map_err(|_| Status::invalid_argument("invalid session id"))?;

        let messages = self
            .state
            .runtime
            .store()
            .list_messages(session_id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let proto_messages: Vec<Message> = messages.iter().map(message_to_proto).collect();

        Ok(Response::new(ListMessagesResponse {
            messages: proto_messages,
        }))
    }
}

fn session_to_proto(s: &crate::store::Session) -> Session {
    Session {
        id: s.id.to_string(),
        title: s.title.clone(),
        model: Some(ModelName {
            value: s.model.as_str().to_owned(),
        }),
        created_at: Some(Timestamp {
            value: s.created_at.to_rfc3339(),
        }),
        updated_at: Some(Timestamp {
            value: s.updated_at.to_rfc3339(),
        }),
        metadata: s.metadata.to_string(),
    }
}

fn message_to_proto(m: &provider::Message) -> Message {
    match m {
        provider::Message::System { content } => Message {
            role: "system".to_owned(),
            content: content.iter().map(content_to_proto).collect(),
            ..Default::default()
        },
        provider::Message::User { content } => Message {
            role: "user".to_owned(),
            content: content.iter().map(content_to_proto).collect(),
            ..Default::default()
        },
        provider::Message::Assistant {
            content,
            tool_calls,
        } => Message {
            role: "assistant".to_owned(),
            content: content.iter().map(content_to_proto).collect(),
            tool_calls: tool_calls.iter().map(tool_call_to_proto).collect(),
            ..Default::default()
        },
        provider::Message::Tool {
            tool_call_id,
            name,
            content,
        } => Message {
            role: "tool".to_owned(),
            content: content.iter().map(content_to_proto).collect(),
            tool_call_id: tool_call_id.clone(),
            tool_name: name.clone(),
            ..Default::default()
        },
    }
}

fn content_to_proto(p: &provider::ContentPart) -> ContentPart {
    match p {
        provider::ContentPart::Text { text } => ContentPart {
            content_type: "text".to_owned(),
            text: text.clone(),
            ..Default::default()
        },
        provider::ContentPart::Json { value } => ContentPart {
            content_type: "json".to_owned(),
            json_value: value.to_string(),
            ..Default::default()
        },
        provider::ContentPart::ImageUrl { url, mime_type } => ContentPart {
            content_type: "image_url".to_owned(),
            image_url: url.clone(),
            mime_type: mime_type.clone().unwrap_or_default(),
            ..Default::default()
        },
    }
}

fn tool_call_to_proto(tc: &provider::ToolCall) -> super::pb::ToolCall {
    super::pb::ToolCall {
        id: tc.id.clone(),
        name: tc.name.clone(),
        arguments: tc.arguments.to_string(),
    }
}
