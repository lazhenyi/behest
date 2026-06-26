//! Demonstrates token estimation utilities.
//!
//! Run with:
//! ```bash
//! cargo run --example token_estimation
//! ```

use behest::provider::{ContentPart, Message, ToolCall};
use behest::token::{
    estimate_content_part_tokens, estimate_message_tokens, estimate_tokens,
    estimate_tool_call_tokens,
};
use serde_json::json;

fn main() {
    let text = "The quick brown fox jumps over the lazy dog.";
    println!("estimate_tokens(\"{text}\") = {}", estimate_tokens(text));

    let chinese = "敏捷的棕色狐狸跳过了懒狗。";
    println!(
        "estimate_tokens(\"{chinese}\") = {}",
        estimate_tokens(chinese)
    );

    let part = ContentPart::text("Hello, world!");
    println!(
        "estimate_content_part_tokens(text) = {}",
        estimate_content_part_tokens(&part)
    );

    let msg = Message::user_text("What is the capital of France?");
    println!(
        "estimate_message_tokens(user) = {}",
        estimate_message_tokens(&msg)
    );

    let sys = Message::system_text("You are a helpful assistant.");
    println!(
        "estimate_message_tokens(system) = {}",
        estimate_message_tokens(&sys)
    );

    let tool_call = ToolCall::new("call_1", "get_weather", json!({"city": "Beijing"}));
    println!(
        "estimate_tool_call_tokens({}) = {}",
        tool_call.name,
        estimate_tool_call_tokens(&tool_call)
    );

    let long = "a".repeat(400);
    println!(
        "estimate_tokens({} chars) = {}",
        long.len(),
        estimate_tokens(&long)
    );
}
