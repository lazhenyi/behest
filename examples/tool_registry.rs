//! Demonstrates creating and registering custom tools with `FunctionTool`.
//!
//! Run with:
//! ```bash
//! cargo run --example tool_registry
//! ```

use behest::tool::{FunctionTool, ToolRegistry};
use serde_json::{Value, json};

fn main() {
    let registry = ToolRegistry::new();

    let add = FunctionTool::new(
        "add",
        "Add two numbers together",
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
        "Echoes the input message",
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

    println!("Registered {} tools", registry.len());

    for name in registry.names() {
        println!("  - {name}");
    }
}
