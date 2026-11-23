//! Length-prefixed frames.
//!
//! Layout (all integers big-endian):
//!
//! ```text
//! frame  = u32 body_len
//!          u8  kind
//!          body[body_len - 1]
//!
//! kind 1 (json)  : body = utf-8 JSON
//! kind 2 (audio) : body = u8 header_len | header_json | pcm_i16_le...
//! kind 3 (close) : body = empty
//! ```
//!
//! `body_len` counts the kind byte plus the body. The reader rejects
//! `body_len > MAX_FRAME_BYTES` **before** allocating, and rejects audio whose
//! PCM payload has an odd byte length.

use std::io;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Hard ceiling for one frame. 1 MiB holds ~32 s of 16 kHz mono PCM, so it is
/// generous for audio and strict enough that a corrupt length cannot exhaust
/// memory.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

const KIND_JSON: u8 = 1;
const KIND_AUDIO: u8 = 2;
const KIND_CLOSE: u8 = 3;

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("peer closed the stream")]
    Closed,
    #[error("frame declares {len} bytes, limit is {max}")]
    TooLarge { len: usize, max: usize },
    #[error("empty frame")]
    Empty,
    #[error("unknown frame kind {0}")]
    UnknownKind(u8),
    #[error("audio header length {0} exceeds body")]
    BadHeaderLen(u8),
    #[error("audio pcm length {0} is not a multiple of 2")]
    OddPcm(usize),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("utf-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioHeader {
    pub request_id: String,
    pub seq: u64,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Json(serde_json::Value),
    Audio { header: AudioHeader, pcm: Vec<i16> },
    Close,
}

/// Read one length-prefixed frame from `r`.
///
/// # Errors
///
/// Returns [`FrameError::Closed`] on a clean EOF, [`FrameError::Io`] for other
/// transport failures, and a protocol-class [`FrameError`] variant when the
/// peer sends an empty, oversized, unknown-kind, malformed-header, odd-PCM,
/// non-UTF-8, or non-JSON frame.
pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R) -> Result<Frame, FrameError> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Err(FrameError::Closed),
        Err(e) => return Err(FrameError::Io(e)),
    }
    let body_len = u32::from_be_bytes(len_buf) as usize;

    if body_len == 0 {
        return Err(FrameError::Empty);
    }
    if body_len > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            len: body_len,
            max: MAX_FRAME_BYTES,
        });
    }

    let mut body = vec![0u8; body_len];
    r.read_exact(&mut body).await?;

    let kind = body[0];
    let payload = &body[1..];
    match kind {
        KIND_JSON => {
            let text = std::str::from_utf8(payload)?;
            Ok(Frame::Json(serde_json::from_str(text)?))
        }
        KIND_AUDIO => {
            let (header_len, rest) = payload.split_first().ok_or(FrameError::BadHeaderLen(0))?;
            // Keep the wire value as `u8` so reporting it never needs a
            // narrowing cast.
            let header_len = *header_len;
            if usize::from(header_len) > rest.len() {
                return Err(FrameError::BadHeaderLen(header_len));
            }
            let (header_bytes, pcm_bytes) = rest.split_at(usize::from(header_len));
            if pcm_bytes.len() % 2 != 0 {
                return Err(FrameError::OddPcm(pcm_bytes.len()));
            }
            let header: AudioHeader = serde_json::from_slice(header_bytes)?;
            let (pairs, _) = pcm_bytes.as_chunks::<2>();
            let pcm = pairs.iter().map(|c| i16::from_le_bytes(*c)).collect();
            Ok(Frame::Audio { header, pcm })
        }
        KIND_CLOSE => Ok(Frame::Close),
        other => Err(FrameError::UnknownKind(other)),
    }
}

/// Write one length-prefixed frame to `w`, flushing afterwards.
///
/// # Errors
///
/// Returns a protocol-class [`FrameError`] when the encoded body exceeds the
/// frame or audio-header limits, and [`FrameError::Io`] when the transport
/// rejects the write or flush.
pub async fn write_frame<W: AsyncWrite + Unpin>(
    w: &mut W,
    frame: &Frame,
) -> Result<(), FrameError> {
    let body = encode_body(frame)?;
    let total_len = body.len() + 1;
    if total_len > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            len: total_len,
            max: MAX_FRAME_BYTES,
        });
    }
    let total = u32::try_from(total_len).map_err(|_| FrameError::TooLarge {
        len: total_len,
        max: MAX_FRAME_BYTES,
    })?;
    let mut out = Vec::with_capacity(4 + total_len);
    out.extend_from_slice(&total.to_be_bytes());
    out.push(match frame {
        Frame::Json(_) => KIND_JSON,
        Frame::Audio { .. } => KIND_AUDIO,
        Frame::Close => KIND_CLOSE,
    });
    out.extend_from_slice(&body);
    w.write_all(&out).await?;
    w.flush().await?;
    Ok(())
}

fn encode_body(frame: &Frame) -> Result<Vec<u8>, FrameError> {
    match frame {
        Frame::Json(v) => Ok(serde_json::to_vec(v)?),
        Frame::Audio { header, pcm } => {
            let header = serde_json::to_vec(header)?;
            let header_len = u8::try_from(header.len()).map_err(|_| FrameError::TooLarge {
                len: header.len(),
                max: usize::from(u8::MAX),
            })?;
            let mut body = Vec::with_capacity(1 + header.len() + pcm.len() * 2);
            body.push(header_len);
            body.extend_from_slice(&header);
            for sample in pcm {
                body.extend_from_slice(&sample.to_le_bytes());
            }
            Ok(body)
        }
        Frame::Close => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn roundtrip(frame: Frame) -> Vec<Frame> {
        let mut buf = Vec::new();
        write_frame(&mut buf, &frame).await.unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded = read_frame(&mut cursor).await.unwrap();
        assert_eq!(decoded, frame);
        let mut buf = Vec::new();
        write_frame(&mut buf, &frame).await.unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        vec![read_frame(&mut cursor).await.unwrap()]
    }

    #[tokio::test]
    async fn json_roundtrip() {
        let v = serde_json::json!({"type": "ready", "plugin": "fake"});
        roundtrip(Frame::Json(v)).await;
    }

    #[tokio::test]
    async fn audio_roundtrip_preserves_samples() {
        let header = AudioHeader {
            request_id: "r1".into(),
            seq: 7,
            sample_rate: 16_000,
            channels: 1,
        };
        roundtrip(Frame::Audio {
            header,
            pcm: vec![1, -2, 3, i16::MIN],
        })
        .await;
    }

    #[tokio::test]
    async fn oversized_length_is_rejected_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(u32::try_from(MAX_FRAME_BYTES).unwrap() + 1).to_be_bytes());
        let mut cursor = std::io::Cursor::new(bytes);
        let err = read_frame(&mut cursor).await.unwrap_err();
        assert!(matches!(err, FrameError::TooLarge { .. }));
    }

    #[tokio::test]
    async fn zero_length_is_rejected() {
        let mut cursor = std::io::Cursor::new(0u32.to_be_bytes().to_vec());
        assert!(matches!(
            read_frame(&mut cursor).await.unwrap_err(),
            FrameError::Empty
        ));
    }

    #[tokio::test]
    async fn odd_pcm_is_rejected() {
        let header = serde_json::to_vec(&AudioHeader {
            request_id: "r".into(),
            seq: 0,
            sample_rate: 16_000,
            channels: 1,
        })
        .unwrap();
        let mut body = vec![u8::try_from(header.len()).unwrap()];
        body.extend_from_slice(&header);
        body.push(0); // one stray byte
        let mut bytes = u32::try_from(body.len() + 1)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        bytes.push(KIND_AUDIO);
        bytes.extend_from_slice(&body);
        let mut cursor = std::io::Cursor::new(bytes);
        assert!(matches!(
            read_frame(&mut cursor).await.unwrap_err(),
            FrameError::OddPcm(1)
        ));
    }

    #[tokio::test]
    async fn eof_is_closed_not_panic() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        assert!(matches!(
            read_frame(&mut cursor).await.unwrap_err(),
            FrameError::Closed
        ));
    }
}
