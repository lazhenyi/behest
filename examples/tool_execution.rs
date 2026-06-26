//! Demonstrates executing tool calls through the `ToolRuntime`.
//!
//! Run with:
//! ```bash
//! cargo run --example tool_execution
//! ```

use behest::provider::ToolCall;
use behest::runtime::{RuntimePolicy, ToolRuntime};
use behest::tool::{FunctionTool, ToolRegistry};
use serde_json::{Value, json};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let registry = ToolRegistry::new();

    let add = FunctionTool::new(
        "add",
        "Add two numbers",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["a", "b"]
        }),
        |args: Value| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({ "result": a + b }))
        },
    );

    let echo = FunctionTool::new(
        "echo",
        "Echoes a message",
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        }),
        |args: Value| async move {
            let msg = args["message"].as_str().unwrap_or("");
            Ok(json!({ "echo": msg }))
        },
    );

    registry.register(add);
    registry.register(echo);

    let tool_runtime = ToolRuntime::new(registry, RuntimePolicy::default());

    let session_id = Uuid::nil();
    let message_id = Uuid::nil();

    let outcomes = tool_runtime
        .execute_batch(
            vec![
                ToolCall::new("call_1", "add", json!({ "a": 3, "b": 4 })),
                ToolCall::new("call_2", "echo", json!({ "message": "hello" })),
            ],
            session_id,
            message_id,
            None::<&dyn behest::store::ExecutionStore>,
        )
        .await
        .map_err(|e| behest::Error::Config(format!("tool execution failed: {e}")))?;

    println!("Executed {} tools", outcomes.len());
    for outcome in &outcomes {
        match &outcome.output {
            Ok(output) => {
                let preview = serde_json::to_string(&output.value).unwrap_or_default();
                println!("  {} => {}", outcome.call.name, preview);
            }
            Err(err) => {
                println!("  {} => error: {err}", outcome.call.name);
            }
        }
    }

    Ok(())
}
