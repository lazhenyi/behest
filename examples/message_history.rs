//! Demonstrates appending messages to a session and listing message history.
//!
//! Run with:
//! ```bash
//! cargo run --example message_history
//! ```

use behest::provider::{ContentPart, ModelName};
use behest::store::memory::MemorySessionStore;
use behest::store::{MessageRecord, MessageRole, Session, SessionStore};

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let store = MemorySessionStore::new();

    let session = Session::new("Demo Chat", ModelName::new("gpt-4o"));
    let session = store
        .create_session(session)
        .await
        .map_err(behest::Error::Storage)?;

    println!("Created session: {}", session.id);

    let user_msg = store
        .append_message(MessageRecord::new(
            session.id,
            MessageRole::User,
            vec![ContentPart::text("What is the capital of France?")],
        ))
        .await
        .map_err(behest::Error::Storage)?;
    println!("Appended user message: {}", user_msg.id);

    let assistant_msg = store
        .append_message(MessageRecord::new(
            session.id,
            MessageRole::Assistant,
            vec![ContentPart::text("The capital of France is Paris.")],
        ))
        .await
        .map_err(behest::Error::Storage)?;
    println!("Appended assistant message: {}", assistant_msg.id);

    let user_msg2 = store
        .append_message(MessageRecord::new(
            session.id,
            MessageRole::User,
            vec![ContentPart::text("What is its population?")],
        ))
        .await
        .map_err(behest::Error::Storage)?;
    println!("Appended user message: {}", user_msg2.id);

    let messages = store
        .list_messages(&session.id)
        .await
        .map_err(behest::Error::Storage)?;

    println!("\nMessage history ({} messages):", messages.len());
    for msg in &messages {
        let text: String = msg
            .content
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => text.clone(),
                _ => "[non-text]".to_string(),
            })
            .collect();
        println!("  [{:?}] {:?}", msg.role, text);
    }

    Ok(())
}
