//! Stream adapters and helpers that enforce the documented streaming contract.

use crate::error::ProviderResult;
use crate::model::ProviderEvent;
use crate::ProviderError;
use futures::stream::Stream;
use futures::StreamExt;
use pin_project::pin_project;
use serde::de::DeserializeOwned;
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::warn;

/// Truncate a streaming payload to a bounded length for safe logging.
///
/// Honors UTF-8 char boundaries so logs never panic on multibyte input.
pub(crate) fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 200;
    if s.len() <= MAX {
        s.to_string()
    } else {
        let mut end = MAX;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

/// Adapter trait for processing SSE events from a provider stream.
///
/// Each provider family implements this to deserialize its specific event
/// format and maintain its own assembly state.
pub(crate) trait SseStreamAdapter: Default {
    /// The provider-specific event type to deserialize from SSE data payloads.
    type Event: DeserializeOwned;

    /// A short label for log messages (e.g. `"Chat Completions"` or `"Anthropic"`).
    const LABEL: &'static str;

    /// Process a deserialized event and return zero or more provider events.
    fn process_event(&mut self, event: Self::Event) -> Vec<ProviderResult<ProviderEvent>>;

    /// Handle the `[DONE]` SSE sentinel. The default implementation does
    /// nothing. Chat Completions overrides this to flush pending tool calls
    /// and emit `Complete`.
    fn handle_done(&mut self) -> Vec<ProviderResult<ProviderEvent>> {
        vec![Ok(ProviderEvent::Complete)]
    }
}

/// Generic SSE stream processor that handles byte fragmentation, SSE framing,
/// deserialization, and error propagation uniformly across provider families.
pub(crate) fn process_sse_stream<A: SseStreamAdapter>(
    response: reqwest::Response,
) -> impl futures::Stream<Item = ProviderResult<ProviderEvent>> {
    let byte_stream = response.bytes_stream();

    let mut sse_parser = crate::sse::SseParser::new();
    let mut adapter = A::default();

    byte_stream.flat_map(move |chunk_result| {
        let mut events: Vec<ProviderResult<ProviderEvent>> = Vec::new();

        match chunk_result {
            Ok(ref bytes) => {
                let sse_events = sse_parser.feed(bytes);

                for sse_event in sse_events {
                    if sse_event.data == "[DONE]" {
                        let done_events = adapter.handle_done();
                        events.extend(done_events);
                        continue;
                    }

                    match serde_json::from_str::<A::Event>(&sse_event.data) {
                        Ok(event) => {
                            let new_events = adapter.process_event(event);
                            events.extend(new_events);
                        }
                        Err(error) => {
                            warn!(
                                error = %error,
                                payload = %truncate_for_log(&sse_event.data),
                                "Failed to parse {} stream event, skipping",
                                A::LABEL
                            );
                        }
                    }
                }
            }
            Err(e) => {
                events.push(Err(ProviderError::transport(e.to_string())));
            }
        }

        futures::stream::iter(events)
    })
}

/// Fuses a provider event stream at the first terminal event.
///
/// A terminal event is any of:
/// - `Err(_)` — transport or protocol error surfaced as an error result.
/// - `Ok(ProviderEvent::Error { .. })` — provider-reported error event.
/// - `Ok(ProviderEvent::Complete)` — successful stream termination.
///
/// The terminal event itself is forwarded to the consumer. Any items
/// produced by the underlying stream after the terminal event are dropped.
/// This enforces the contract that `Complete` cannot follow `Error` and
/// that no further events are emitted after a stream has terminated.
#[pin_project]
pub(crate) struct TerminatingStream<S> {
    #[pin]
    inner: S,
    terminated: bool,
}

impl<S> TerminatingStream<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner,
            terminated: false,
        }
    }
}

impl<S> Stream for TerminatingStream<S>
where
    S: Stream<Item = ProviderResult<ProviderEvent>>,
{
    type Item = ProviderResult<ProviderEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        if *this.terminated {
            return Poll::Ready(None);
        }
        match this.inner.poll_next(cx) {
            Poll::Ready(Some(item)) => {
                if is_terminal(&item) {
                    *this.terminated = true;
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                *this.terminated = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn is_terminal(item: &ProviderResult<ProviderEvent>) -> bool {
    matches!(
        item,
        Err(_) | Ok(ProviderEvent::Complete) | Ok(ProviderEvent::Error { .. })
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProviderError;
    use futures::stream::{self, StreamExt};

    async fn collect<S>(stream: S) -> Vec<ProviderResult<ProviderEvent>>
    where
        S: Stream<Item = ProviderResult<ProviderEvent>>,
    {
        stream.collect().await
    }

    #[tokio::test]
    async fn drops_events_after_complete() {
        let items: Vec<ProviderResult<ProviderEvent>> = vec![
            Ok(ProviderEvent::Output {
                content: "hi".into(),
            }),
            Ok(ProviderEvent::Complete),
            Ok(ProviderEvent::Output {
                content: "after".into(),
            }),
            Ok(ProviderEvent::Complete),
        ];
        let out = collect(TerminatingStream::new(stream::iter(items))).await;
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Ok(ProviderEvent::Output { .. })));
        assert!(matches!(out[1], Ok(ProviderEvent::Complete)));
    }

    #[tokio::test]
    async fn drops_complete_after_error_event() {
        let items: Vec<ProviderResult<ProviderEvent>> = vec![
            Ok(ProviderEvent::Output {
                content: "hi".into(),
            }),
            Ok(ProviderEvent::Error {
                source: crate::ProviderError::general("boom"),
            }),
            Ok(ProviderEvent::Complete),
        ];
        let out = collect(TerminatingStream::new(stream::iter(items))).await;
        assert_eq!(out.len(), 2);
        assert!(matches!(out[1], Ok(ProviderEvent::Error { .. })));
    }

    #[tokio::test]
    async fn drops_complete_after_err_result() {
        let items: Vec<ProviderResult<ProviderEvent>> = vec![
            Ok(ProviderEvent::Output {
                content: "hi".into(),
            }),
            Err(ProviderError::transport("connection reset")),
            Ok(ProviderEvent::Complete),
        ];
        let out = collect(TerminatingStream::new(stream::iter(items))).await;
        assert_eq!(out.len(), 2);
        assert!(out[1].is_err());
    }

    #[tokio::test]
    async fn passes_through_non_terminal_events() {
        let items: Vec<ProviderResult<ProviderEvent>> = vec![
            Ok(ProviderEvent::Status {
                message: "thinking".into(),
            }),
            Ok(ProviderEvent::Output {
                content: "hi".into(),
            }),
            Ok(ProviderEvent::Complete),
        ];
        let out = collect(TerminatingStream::new(stream::iter(items))).await;
        assert_eq!(out.len(), 3);
    }
}
