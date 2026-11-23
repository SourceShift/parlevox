//! The `openai-audio-tts` connector.
//!
//! Requests raw little-endian PCM and assembles it into fixed 20 ms frames at
//! the requested sample rate and channel count. Frames wait in a bounded queue
//! so a burst from the network cannot grow memory without limit; anything the
//! queue discards is counted and reported through
//! `take_sidecar_dropped_frames`.

use std::sync::Arc;

use async_trait::async_trait;
use parlevox_core::ParlevoxError;
use parlevox_providers::{
    AudioChunk, ProviderManifest, ProviderStream, SynthesizeRequest, TtsProvider,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::buffer::BoundedQueue;
use crate::buffer::ConnectorStats;
use crate::error::ConnectorError;
use crate::stream::BodySource;
use crate::toolkit::HttpToolkit;
use crate::transport::RequestSpec;

/// Frame duration, in milliseconds, of each emitted [`AudioChunk`].
const FRAME_MS: u64 = 20;

/// The `openai-audio-tts` connector.
pub struct OpenAiTts {
    manifest: ProviderManifest,
    endpoint: String,
    model: String,
    voice: String,
    max_buffered_frames: usize,
    toolkit: HttpToolkit,
}

impl OpenAiTts {
    pub(crate) fn new(
        manifest: ProviderManifest,
        endpoint: String,
        model: String,
        voice: String,
        max_buffered_frames: usize,
        toolkit: HttpToolkit,
    ) -> Self {
        OpenAiTts {
            manifest,
            endpoint,
            model,
            voice,
            max_buffered_frames,
            toolkit,
        }
    }

    fn validate_request(&self, request: &SynthesizeRequest) -> Result<(), ParlevoxError> {
        if request.text.trim().is_empty() {
            return Err(ParlevoxError::Protocol(format!(
                "provider '{}' cannot synthesize empty text",
                self.manifest.name
            )));
        }
        if request.sample_rate == 0 || request.channels == 0 {
            return Err(ParlevoxError::Protocol(format!(
                "provider '{}' received an invalid audio format",
                self.manifest.name
            )));
        }
        if let Some(model) = self
            .manifest
            .models
            .iter()
            .find(|model| model.id == self.model)
        {
            if !model.limits.sample_rates.is_empty()
                && !model.limits.sample_rates.contains(&request.sample_rate)
            {
                return Err(ParlevoxError::Protocol(format!(
                    "model '{}' does not support sample rate {}",
                    self.model, request.sample_rate
                )));
            }
            if !model.limits.channels.is_empty()
                && !model.limits.channels.contains(&request.channels)
            {
                return Err(ParlevoxError::Protocol(format!(
                    "model '{}' does not support {} channels",
                    self.model, request.channels
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl TtsProvider for OpenAiTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        self.validate_request(&request)?;
        let provider = self.manifest.name.clone();
        let body = json!({
            "model": self.model,
            "input": request.text,
            "voice": self.voice,
            "response_format": "pcm",
            "sample_rate": request.sample_rate,
        });
        let spec = RequestSpec::post(&self.endpoint)
            .with_header("accept", "audio/pcm")
            .map_err(|error| error.into_provider(&provider))?
            .with_json(body);
        let response = self
            .toolkit
            .send(spec, &cancel)
            .await
            .map_err(|error| error.into_provider(&provider))?;
        let body = response.into_body_source(cancel);
        Ok(pcm_stream(
            body,
            request.sample_rate,
            request.channels,
            &self.toolkit,
            self.max_buffered_frames,
            provider,
        ))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.toolkit.stats().take_dropped()
    }
}

enum PcmStatus {
    /// Reading the body.
    Open,
    /// Body finished; buffered frames remain to be delivered.
    Draining,
    /// A failure ended the stream; nothing more is emitted.
    Failed,
}

struct PcmState {
    body: BodySource,
    assembler: PcmAssembler,
    queue: BoundedQueue<Vec<i16>>,
    stats: Arc<ConnectorStats>,
    sample_rate: u32,
    channels: u16,
    provider: String,
    status: PcmStatus,
}

impl PcmState {
    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, ConnectorError> {
        loop {
            if matches!(self.status, PcmStatus::Failed) {
                return Ok(None);
            }
            if let Some(frame) = self.queue.pop() {
                self.stats.record_delivered(1);
                return Ok(Some(AudioChunk::new(
                    self.sample_rate,
                    self.channels,
                    frame,
                )));
            }
            if matches!(self.status, PcmStatus::Draining) {
                return Ok(None);
            }
            match self.body.next().await {
                None => {
                    self.status = PcmStatus::Draining;
                }
                Some(Err(error)) => {
                    self.status = PcmStatus::Failed;
                    return Err(error);
                }
                Some(Ok(chunk)) => {
                    let frames = self.assembler.push(&chunk)?;
                    self.push_frames(frames);
                }
            }
        }
    }

    fn push_frames(&mut self, frames: Vec<Vec<i16>>) {
        if frames.is_empty() {
            return;
        }
        self.stats.record_pushed(frames.len());
        let dropped = self.queue.push_all(frames);
        if dropped != 0 {
            self.stats.record_dropped(dropped);
        }
    }
}

fn pcm_stream(
    body: BodySource,
    sample_rate: u32,
    channels: u16,
    toolkit: &HttpToolkit,
    max_buffered_frames: usize,
    provider: String,
) -> ProviderStream<AudioChunk> {
    let assembler = PcmAssembler::new(sample_rate, channels);
    let state = PcmState {
        body,
        assembler,
        queue: BoundedQueue::new(max_buffered_frames),
        stats: Arc::clone(toolkit.stats()),
        sample_rate,
        channels,
        provider,
        status: PcmStatus::Open,
    };
    Box::pin(futures_util::stream::unfold(
        state,
        |mut state| async move {
            match state.next_chunk().await {
                Ok(Some(chunk)) => Some((Ok(chunk), state)),
                Ok(None) => None,
                Err(error) => {
                    let mapped = error.into_provider(&state.provider);
                    state.status = PcmStatus::Failed;
                    Some((Err(mapped), state))
                }
            }
        },
    ))
}

/// Splits a little-endian PCM byte stream into fixed-duration frames.
struct PcmAssembler {
    bytes: Vec<u8>,
    frame_bytes: usize,
}

impl PcmAssembler {
    fn new(sample_rate: u32, channels: u16) -> Self {
        let per_channel = (u64::from(sample_rate).saturating_mul(FRAME_MS) / 1000).max(1);
        let frame_bytes = per_channel
            .saturating_mul(u64::from(channels))
            .saturating_mul(2);
        let frame_bytes = usize::try_from(frame_bytes).unwrap_or(usize::MAX).max(2);
        PcmAssembler {
            bytes: Vec::new(),
            frame_bytes,
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Result<Vec<Vec<i16>>, ConnectorError> {
        self.bytes.extend_from_slice(chunk);
        let complete = self.bytes.len() / self.frame_bytes;
        let mut frames = Vec::with_capacity(complete);
        for _ in 0..complete {
            let frame: Vec<u8> = self.bytes.drain(..self.frame_bytes).collect();
            let mut pcm = Vec::with_capacity(frame.len() / 2);
            for pair in frame.chunks_exact(2) {
                let sample: [u8; 2] = pair
                    .try_into()
                    .map_err(|_| ConnectorError::Protocol("incomplete PCM sample".to_owned()))?;
                pcm.push(i16::from_le_bytes(sample));
            }
            frames.push(pcm);
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_fixed_frames_and_buffers_the_tail() {
        let mut assembler = PcmAssembler::new(1000, 1);
        // 20 samples per frame at 1 kHz / 20 ms, 2 bytes each.
        assert_eq!(assembler.frame_bytes, 40);
        let frames = assembler
            .push(&[0u8; 100])
            .expect("two complete frames and a half frame");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].len(), 20);
        assert_eq!(assembler.bytes.len(), 20);
    }

    #[test]
    fn sample_values_round_trip_little_endian() {
        let mut assembler = PcmAssembler::new(1000, 1);
        let mut bytes = Vec::new();
        for value in 0i16..20 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let frames = assembler.push(&bytes).expect("one frame");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0][0], 0);
        assert_eq!(frames[0][19], 19);
    }
}
