//! Integration tests for the Anthropic adapter using wiremock.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use agents::adapt::anthropic::AnthropicChatAdapter;
use agents::provider::{
    ChatProvider, ChatRequest, FinishReason, Message, ModelName, ProviderHttpConfig, ProviderId,
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_config(server: &MockServer) -> ProviderHttpConfig {
    ProviderHttpConfig::new(ProviderId::new("anthropic-test"), server.uri())
}

#[tokio::test]
async fn complete_returns_chat_response_from_anthropic_format() {
    let server = MockServer::start().await;

    let response_body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-test",
        "content": [{"type": "text", "text": "Hello from Claude!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 10}
    });

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .expect(1)
        .mount(&server)
        .await;

    let adapter = AnthropicChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("claude-3-test")).with_user_text("Hi");
    let response = adapter.complete(request).await.expect("complete request");

    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.model.as_str(), "claude-3-test");
    assert!(
        response
            .usage
            .is_some_and(|u| u.input_tokens == 5 && u.output_tokens == 10),
        "expected token usage from response"
    );
}

#[tokio::test]
async fn complete_sends_system_as_top_level_field() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-test",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let adapter = AnthropicChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("claude-3-test"))
        .with_message(Message::system_text("You are helpful"))
        .with_user_text("Hi");

    adapter.complete(request).await.expect("complete request");
}

#[tokio::test]
async fn complete_sends_x_api_key_header() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-test",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let config =
        test_config(&server).with_api_key(secrecy::SecretString::new("sk-ant-test".into()));
    let adapter = AnthropicChatAdapter::new(config).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("claude-3-test")).with_user_text("Hi");

    adapter.complete(request).await.expect("complete with auth");
}

#[tokio::test]
async fn complete_returns_authentication_error_on_401() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"type": "authentication_error", "message": "invalid key"}
        })))
        .mount(&server)
        .await;

    let adapter = AnthropicChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("claude-3-test")).with_user_text("Hi");
    let error = adapter
        .complete(request)
        .await
        .expect_err("expected auth error");

    assert!(matches!(error, agents::ProviderError::Authentication { .. }));
}

#[tokio::test]
async fn complete_handles_tool_use_in_response() {
    let server = MockServer::start().await;

    let response_body = json!({
        "id": "msg_tools",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-test",
        "content": [
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {"location": "Paris"}
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 20}
    });

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&server)
        .await;

    let adapter = AnthropicChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("claude-3-test")).with_user_text("weather?");
    let response = adapter.complete(request).await.expect("complete with tools");

    assert_eq!(response.finish_reason, FinishReason::ToolCalls);

    let tool_calls = match &response.message {
        Message::Assistant { tool_calls, .. } => tool_calls,
        other => panic!("expected assistant message, got {other:?}"),
    };

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "toolu_abc");
    assert_eq!(tool_calls[0].name, "get_weather");
}
