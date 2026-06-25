//! AgentService gRPC implementation.
//!
//! Provides RPCs for registering, querying, listing agents,
//! and setting the default agent for new sessions.

use tonic::{Request, Response, Status};

use crate::agent::{AgentDefinition, AgentMode, PermissionEffect, PermissionRule};
use crate::grpc::pb::{
    AgentDefinition as PbAgentDefinition, AgentMode as PbAgentMode, GetAgentRequest,
    GetAgentResponse, ListAgentsRequest, ListAgentsResponse,
    PermissionEffect as PbPermissionEffect, PermissionRule as PbPermissionRule,
    RegisterAgentRequest, RegisterAgentResponse, SetDefaultAgentRequest, SetDefaultAgentResponse,
    agent_service_server::AgentService,
};
use crate::provider::ModelName;

use std::sync::Arc;

/// gRPC agent service for managing agent definitions.
///
/// Supports registration, lookup, listing, and default agent
/// selection for the runtime.
pub struct GrpcAgentService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcAgentService {
    /// Creates a new agent service backed by the given shared state.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl AgentService for GrpcAgentService {
    async fn register_agent(
        &self,
        request: Request<RegisterAgentRequest>,
    ) -> Result<Response<RegisterAgentResponse>, Status> {
        let req = request.into_inner();

        let Some(pb_agent) = req.agent else {
            return Err(Status::invalid_argument("agent definition is required"));
        };

        let agent = agent_definition_from_proto(pb_agent)?;

        let mut registry = self.state.agent_registry.write().await;
        registry.register(agent);

        Ok(Response::new(RegisterAgentResponse {}))
    }

    async fn get_agent(
        &self,
        request: Request<GetAgentRequest>,
    ) -> Result<Response<GetAgentResponse>, Status> {
        let req = request.into_inner();

        let registry = self.state.agent_registry.read().await;
        let agent = registry
            .get(&req.name)
            .ok_or_else(|| Status::not_found(format!("agent '{}' not found", req.name)))?;

        Ok(Response::new(GetAgentResponse {
            agent: Some(agent_definition_to_proto(agent)),
        }))
    }

    async fn list_agents(
        &self,
        _request: Request<ListAgentsRequest>,
    ) -> Result<Response<ListAgentsResponse>, Status> {
        let registry = self.state.agent_registry.read().await;

        let agents: Vec<PbAgentDefinition> = registry
            .list()
            .into_iter()
            .map(agent_definition_to_proto)
            .collect();
        let default_agent = registry
            .default_agent()
            .map_or_else(String::new, |a| a.name.clone());

        Ok(Response::new(ListAgentsResponse {
            agents,
            default_agent,
        }))
    }

    async fn set_default_agent(
        &self,
        request: Request<SetDefaultAgentRequest>,
    ) -> Result<Response<SetDefaultAgentResponse>, Status> {
        let req = request.into_inner();

        let mut registry = self.state.agent_registry.write().await;

        if registry.get(&req.name).is_none() {
            return Err(Status::not_found(format!("agent '{}' not found", req.name)));
        }

        registry.set_default(req.name);

        Ok(Response::new(SetDefaultAgentResponse {}))
    }
}

#[allow(clippy::result_large_err)]
fn agent_definition_from_proto(pb: PbAgentDefinition) -> Result<AgentDefinition, Status> {
    let mode = match PbAgentMode::try_from(pb.mode) {
        Ok(PbAgentMode::Subagent) => AgentMode::Subagent,
        _ => AgentMode::Primary,
    };

    let permissions: Vec<PermissionRule> = pb
        .permissions
        .into_iter()
        .map(|p| {
            let effect =
                p.effect
                    .and_then(|e| e.effect)
                    .map_or(PermissionEffect::Ask, |e| match e {
                        crate::grpc::pb::permission_effect::Effect::Allow(_) => {
                            PermissionEffect::Allow
                        }
                        crate::grpc::pb::permission_effect::Effect::Deny(_) => {
                            PermissionEffect::Deny
                        }
                        crate::grpc::pb::permission_effect::Effect::Ask(_) => PermissionEffect::Ask,
                    });

            PermissionRule {
                tool: p.tool,
                resource: p.resource,
                effect,
            }
        })
        .collect();

    let model = if pb.model.is_some() {
        pb.model.map(|m| ModelName::new(m.value))
    } else {
        None
    };

    let max_steps = if pb.max_steps == 0 {
        None
    } else {
        Some(pb.max_steps as usize)
    };

    Ok(AgentDefinition {
        name: pb.name,
        description: pb.description,
        mode,
        system_prompt: pb.system_prompt,
        hidden: pb.hidden,
        permissions,
        model,
        max_steps,
    })
}

fn agent_definition_to_proto(agent: &AgentDefinition) -> PbAgentDefinition {
    let mode = match agent.mode {
        AgentMode::Primary => PbAgentMode::Primary as i32,
        AgentMode::Subagent => PbAgentMode::Subagent as i32,
    };

    let permissions: Vec<PbPermissionRule> = agent
        .permissions
        .iter()
        .map(|p| {
            let effect = match p.effect {
                PermissionEffect::Allow => Some(PbPermissionEffect {
                    effect: Some(crate::grpc::pb::permission_effect::Effect::Allow(true)),
                }),
                PermissionEffect::Deny => Some(PbPermissionEffect {
                    effect: Some(crate::grpc::pb::permission_effect::Effect::Deny(true)),
                }),
                PermissionEffect::Ask => Some(PbPermissionEffect {
                    effect: Some(crate::grpc::pb::permission_effect::Effect::Ask(true)),
                }),
            };

            PbPermissionRule {
                tool: p.tool.clone(),
                resource: p.resource.clone(),
                effect,
            }
        })
        .collect();

    let model = agent.model.as_ref().map(|m| crate::grpc::pb::ModelName {
        value: m.as_str().to_string(),
    });

    let max_steps = agent
        .max_steps
        .map_or(0, |s| u32::try_from(s).unwrap_or(u32::MAX));

    PbAgentDefinition {
        name: agent.name.clone(),
        description: agent.description.clone(),
        mode,
        system_prompt: agent.system_prompt.clone(),
        hidden: agent.hidden,
        permissions,
        model,
        max_steps,
    }
}
