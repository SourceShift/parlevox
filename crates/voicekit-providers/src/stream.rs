//! Small stream helpers shared by the fakes and by tests.
//!
//! `futures-util` is a dev-dependency only, so the library side builds streams
//! from a vector and polls the cancellation token itself.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use tokio_util::sync::CancellationToken;
use voicekit_core::VoicekitError;

use crate::traits::ProviderStream;

/// A stream over pre-computed items that stops on cancellation.
///
/// When the token fires the stream yields one [`VoicekitError::Cancelled`] and
/// then ends, so callers can distinguish "provider finished" from "we asked it
/// to stop" without consulting the token themselves.
pub struct CancelableStream<T> {
    items: VecDeque<Result<T, VoicekitError>>,
    cancel: CancellationToken,
    cancel_reported: bool,
    done: bool,
}

impl<T: Unpin> Stream for CancelableStream<T> {
    type Item = Result<T, VoicekitError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if self.cancel.is_cancelled() {
            if self.cancel_reported {
                self.done = true;
                return Poll::Ready(None);
            }
            self.cancel_reported = true;
            return Poll::Ready(Some(Err(VoicekitError::Cancelled)));
        }
        match self.items.pop_front() {
            Some(item) => Poll::Ready(Some(item)),
            None => {
                self.done = true;
                Poll::Ready(None)
            }
        }
    }
}

/// Box a finite, deterministic sequence as a provider stream.
pub fn cancelable_stream<T: Unpin + Send + 'static>(
    items: Vec<Result<T, VoicekitError>>,
    cancel: CancellationToken,
) -> ProviderStream<T> {
    Box::pin(CancelableStream {
        items: items.into(),
        cancel,
        cancel_reported: false,
        done: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn stream_ends_after_items() {
        let cancel = CancellationToken::new();
        let mut s = cancelable_stream(vec![Ok(1), Ok(2), Ok(3)], cancel);
        let mut seen = Vec::new();
        while let Some(item) = s.next().await {
            seen.push(item.unwrap());
        }
        assert_eq!(seen, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn cancellation_yields_one_error_then_ends() {
        let cancel = CancellationToken::new();
        let mut s = cancelable_stream(vec![Ok(1), Ok(2)], cancel.clone());
        assert_eq!(s.next().await.unwrap().unwrap(), 1);
        cancel.cancel();
        let err = s.next().await.unwrap().unwrap_err();
        assert!(matches!(err, VoicekitError::Cancelled));
        assert!(s.next().await.is_none());
    }
}
