//! Integration tests for the OpenAI adapter using wiremock.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use agents::adapt::openai::OpenAiChatAdapter;
use agents::provider::{
    ChatProvider, ChatRequest, FinishReason, ModelName, ProviderHttpConfig, ProviderId,
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_config(server: &MockServer) -> ProviderHttpConfig {
    ProviderHttpConfig::new(ProviderId::new("openai-test"), server.uri())
}

#[tokio::test]
async fn complete_returns_chat_response_from_openai_format() {
    let server = MockServer::start().await;

    let response_body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4-test",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from OpenAI!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15
        }
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .expect(1)
        .mount(&server)
        .await;

    let adapter = OpenAiChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("gpt-4-test")).with_user_text("Hi");
    let response = adapter.complete(request).await.expect("complete request");

    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.model.as_str(), "gpt-4-test");
    assert!(
        response
            .usage
            .is_some_and(|u| u.input_tokens == 5 && u.output_tokens == 10),
        "expected token usage from response"
    );
}

#[tokio::test]
async fn complete_returns_authentication_error_on_401() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"message": "invalid api key"}
        })))
        .mount(&server)
        .await;

    let adapter = OpenAiChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("gpt-4-test")).with_user_text("Hi");
    let error = adapter
        .complete(request)
        .await
        .expect_err("expected auth error");

    assert!(matches!(error, agents::ProviderError::Authentication { .. }));
}

#[tokio::test]
async fn complete_returns_rate_limited_error_on_429() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"message": "rate limited"}
        })))
        .mount(&server)
        .await;

    let adapter = OpenAiChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("gpt-4-test")).with_user_text("Hi");
    let error = adapter
        .complete(request)
        .await
        .expect_err("expected rate limit error");

    assert!(matches!(error, agents::ProviderError::RateLimited { .. }));
    assert!(error.is_retryable());
}

#[tokio::test]
async fn complete_sends_bearer_auth_header_when_key_configured() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "model": "gpt-4-test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let config = test_config(&server).with_api_key(secrecy::SecretString::new("test-key".into()));
    let adapter = OpenAiChatAdapter::new(config).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("gpt-4-test")).with_user_text("Hi");

    adapter.complete(request).await.expect("complete with auth");
}

#[tokio::test]
async fn complete_handles_tool_calls_in_response() {
    let server = MockServer::start().await;

    let response_body = json!({
        "id": "chatcmpl-tools",
        "model": "gpt-4-test",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Paris\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&server)
        .await;

    let adapter = OpenAiChatAdapter::new(test_config(&server)).expect("build adapter");
    let request = ChatRequest::new(ModelName::new("gpt-4-test")).with_user_text("weather?");
    let response = adapter.complete(request).await.expect("complete with tools");

    assert_eq!(response.finish_reason, FinishReason::ToolCalls);

    let tool_calls = match &response.message {
        agents::provider::Message::Assistant { tool_calls, .. } => tool_calls,
        other => panic!("expected assistant message, got {other:?}"),
    };

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_abc");
    assert_eq!(tool_calls[0].name, "get_weather");
}
