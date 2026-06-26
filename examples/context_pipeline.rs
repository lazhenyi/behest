//! Demonstrates composing context adapters with `ContextFactory`.
//!
//! Run with:
//! ```bash
//! cargo run --example context_pipeline
//! ```

use behest::context::{ContextFactory, ContextInput, FunctionAdapter, StaticAdapter};
use behest::provider::{ContentPart, Message, ModelName};

#[tokio::main]
async fn main() {
    let mut factory = ContextFactory::new();

    factory.register(StaticAdapter::system(
        "You are a helpful assistant with access to tools.",
    ));

    factory.register(FunctionAdapter::new(
        "greeter",
        |input: ContextInput| async move {
            let user = input.user_message.unwrap_or_else(|| "anonymous".into());
            Ok(vec![Message::user_text(format!("Hello, {user}!"))])
        },
    ));

    println!("Registered {} adapters: ", factory.len());
    for name in factory.adapter_names() {
        println!("  - {name}");
    }

    let input = ContextInput::new().with_user_message("Alice");
    let output = match factory.build(&input).await {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Context build failed: {e}");
            return;
        }
    };
    println!("\nContext built: {} messages", output.messages().len());
    for msg in output.messages() {
        let role = match msg {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tool { .. } => "tool",
            _ => "unknown",
        };
        let preview: String = match msg {
            Message::System { content } | Message::User { content } => content
                .first()
                .map(|p| match p {
                    ContentPart::Text { text } => text.clone(),
                    _ => String::new(),
                })
                .unwrap_or_default(),
            _ => String::new(),
        };
        let truncated = if preview.len() > 60 {
            format!("{}...", &preview[..60])
        } else {
            preview
        };
        println!("  [{role}] {truncated}");
    }

    let request = output.into_request(ModelName::new("gpt-4o"));
    println!("\nChat request:");
    println!("  model:     {}", request.model);
    println!("  messages:  {}", request.messages.len());
    println!("  tools:     {}", request.tools.len());
}
