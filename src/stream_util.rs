//! Stream adapters that enforce the documented streaming contract.

use crate::error::ProviderResult;
use crate::model::ProviderEvent;
use futures::stream::Stream;
use pin_project::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};

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
                message: "boom".into(),
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
