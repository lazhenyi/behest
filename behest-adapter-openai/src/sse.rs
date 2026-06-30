//! Minimal Server-Sent Events line parser for provider streaming responses.

use bytes::Bytes;
use futures_util::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

use behest_core::error::ProviderError;
use behest_core::error::TransportError;
use behest_provider::ProviderId;

/// A single parsed SSE event with optional event name and data payload.
///
/// Constructed by [`SseStream`] from raw byte chunks. The `event` field captures
/// the `event:` line and the `data` field concatenates all `data:` lines with
/// newline separators.
#[derive(Debug, Clone)]
pub(crate) struct SseEvent {
    /// Named event type from the `event:` field, if present.
    #[allow(dead_code)]
    pub(crate) event: Option<String>,
    /// Concatenated `data:` field value.
    pub(crate) data: String,
}

impl SseEvent {
    /// Returns `true` when this event signals the end of an OpenAI stream.
    ///
    /// OpenAI uses a bare `data: [DONE]` line to mark the end of a stream.
    #[must_use]
    pub(crate) fn is_openai_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }
}

/// Parses a byte stream into SSE events.
///
/// Wraps an inner byte stream and buffers partial chunks until a complete
/// SSE event boundary (`\n\n`, `\r\n\r\n`, or `\r\r`) is found.
///
/// # Type parameters
///
/// * `S` — The inner byte stream, expected to yield `Result<Bytes, reqwest::Error>`.
pub(crate) struct SseStream<S> {
    inner: S,
    buffer: String,
    provider: ProviderId,
}

impl<S> SseStream<S> {
    /// Creates an SSE parser wrapping a byte stream.
    pub(crate) fn new(inner: S, provider: ProviderId) -> Self {
        Self {
            inner,
            buffer: String::new(),
            provider,
        }
    }
}

impl<S> Stream for SseStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<SseEvent, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if let Some(event) = extract_event(&mut this.buffer) {
                return Poll::Ready(Some(Ok(event)));
            }

            let pinned = Pin::new(&mut this.inner);
            match pinned.poll_next(context) {
                Poll::Ready(Some(Ok(chunk))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&chunk));
                }
                Poll::Ready(Some(Err(source))) => {
                    return Poll::Ready(Some(Err(ProviderError::Transport {
                        provider: this.provider.clone(),
                        source: TransportError::new(source),
                    })));
                }
                Poll::Ready(None) => {
                    if let Some(event) = extract_event(&mut this.buffer) {
                        return Poll::Ready(Some(Ok(event)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn extract_event(buffer: &mut String) -> Option<SseEvent> {
    let boundary = find_event_boundary(buffer)?;

    let raw: String = buffer.drain(..boundary.end).collect();
    parse_event_block(&raw)
}

/// Locates the first SSE event boundary (double newline) in the buffer.
///
/// The SSE specification allows `\n\n`, `\r\n\r\n`, and `\r\r` as event
/// delimiters. Returns the byte range to drain (including the delimiter).
fn find_event_boundary(buffer: &str) -> Option<EventBoundary> {
    let bytes = buffer.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        match bytes[i] {
            b'\n' if bytes[i + 1] == b'\n' => {
                return Some(EventBoundary { end: i + 2 });
            }
            b'\r'
                if i + 3 < bytes.len()
                    && bytes[i + 1] == b'\n'
                    && bytes[i + 2] == b'\r'
                    && bytes[i + 3] == b'\n' =>
            {
                return Some(EventBoundary { end: i + 4 });
            }
            b'\r' if bytes[i + 1] == b'\r' => {
                return Some(EventBoundary { end: i + 2 });
            }
            _ => {}
        }
    }
    None
}

/// Byte offset marking the end of an SSE event delimiter.
struct EventBoundary {
    end: usize,
}

/// Parses a raw event block (drained from the buffer) into an [`SseEvent`].
///
/// Scans for `event:` and `data:` lines. Returns `None` when no `data:` lines
/// are present (e.g. empty or comment-only blocks).
fn parse_event_block(block: &str) -> Option<SseEvent> {
    let mut event_name: Option<String> = None;
    let mut data_lines: Vec<String> = Vec::new();

    for line in block.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            event_name = Some(value.trim().to_owned());
        } else if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.trim_start().to_owned());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    Some(SseEvent {
        event: event_name,
        data: data_lines.join("\n"),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn find_event_boundary_accepts_lf_lf() {
        let boundary = find_event_boundary("data: hello\n\n").expect("boundary");
        assert_eq!(boundary.end, "data: hello\n\n".len());
    }

    #[test]
    fn find_event_boundary_accepts_crlf_crlf() {
        let boundary = find_event_boundary("data: hello\r\n\r\n").expect("boundary");
        assert_eq!(boundary.end, "data: hello\r\n\r\n".len());
    }

    #[test]
    fn find_event_boundary_accepts_cr_cr() {
        let boundary = find_event_boundary("data: hello\r\r").expect("boundary");
        assert_eq!(boundary.end, "data: hello\r\r".len());
    }
}
