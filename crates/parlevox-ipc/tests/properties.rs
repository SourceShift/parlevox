//! Property checks for the framed protocol: round-trips preserve data and
//! malformed frames are rejected rather than silently truncated.

use std::io::Cursor;
use std::sync::OnceLock;

use parlevox_ipc::{read_frame, write_frame, AudioHeader, Frame, FrameError, MAX_FRAME_BYTES};
use proptest::prelude::*;
use tokio::runtime::Runtime;

const KIND_AUDIO: u8 = 2;

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME
        .get_or_init(|| Runtime::new().expect("test runtime"))
        .block_on(future)
}

async fn roundtrip(frame: Frame) -> Frame {
    let mut buf = Vec::new();
    write_frame(&mut buf, &frame).await.expect("encode");
    let mut cursor = Cursor::new(buf);
    read_frame(&mut cursor).await.expect("decode")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn json_frames_roundtrip(values in prop::collection::vec(any::<i64>(), 0..8)) {
        let frame = Frame::Json(serde_json::json!({"values": values}));
        let decoded = block_on(roundtrip(frame.clone()));
        prop_assert_eq!(decoded, frame);
    }

    #[test]
    fn audio_frames_roundtrip(
        pcm in prop::collection::vec(any::<i16>(), 0..256),
        sample_rate in 8_000u32..48_000u32,
        channels in 1u16..3u16,
        seq in any::<u64>(),
    ) {
        let frame = Frame::Audio {
            header: AudioHeader {
                request_id: "r-prop".into(),
                seq,
                sample_rate,
                channels,
            },
            pcm,
        };
        let decoded = block_on(roundtrip(frame.clone()));
        prop_assert_eq!(decoded, frame);
    }

    #[test]
    fn oversized_declared_length_is_rejected_before_allocation(
        length in (MAX_FRAME_BYTES as u64 + 1)..=(u32::MAX as u64),
    ) {
        let bytes = (length as u32).to_be_bytes().to_vec();
        let mut cursor = Cursor::new(bytes);
        let result = block_on(read_frame(&mut cursor));
        let too_large = matches!(result, Err(FrameError::TooLarge { .. }));
        prop_assert!(too_large);
        // Only the four length bytes were consumed.
        prop_assert_eq!(cursor.position(), 4);
    }

    #[test]
    fn odd_pcm_is_rejected(odd_len in (1usize..64).prop_map(|n| n * 2 + 1)) {
        let header = serde_json::to_vec(&AudioHeader {
            request_id: "r".into(),
            seq: 0,
            sample_rate: 16_000,
            channels: 1,
        })
        .unwrap();
        let mut body = vec![header.len() as u8];
        body.extend_from_slice(&header);
        body.extend(std::iter::repeat_n(0u8, odd_len));
        let mut bytes = ((body.len() + 1) as u32).to_be_bytes().to_vec();
        bytes.push(KIND_AUDIO);
        bytes.extend_from_slice(&body);
        let mut cursor = Cursor::new(bytes);
        let result = block_on(read_frame(&mut cursor));
        let odd_rejected = matches!(result, Err(FrameError::OddPcm(n)) if n == odd_len);
        prop_assert!(odd_rejected);
    }
}
