//! Agorabus topic + payload schema for `wm-stt`.
//!
//! The daemon subscribes to `wm.audio.speech.*` (published by `wm-audio`)
//! and publishes its own `wm.stt.*` events; an optional command topic
//! `wm.stt.reload_model` triggers a hot model swap. Payloads round-trip
//! through `serde_json::Value` because the agorabus
//! [`ServerEvent::data`](agorabus::ServerEvent) is a `Value` — the daemon
//! [`crate::daemon`] (iter-4+) decodes per-topic into the request enums
//! defined here.

use serde::{Deserialize, Serialize};

/// Subscribe prefix that captures inbound speech segments from `wm-audio`.
pub const AUDIO_TOPIC_PREFIX: &str = "wm.audio.speech.";

/// Subscribe prefix that captures inbound `wm-stt` command topics.
pub const STT_COMMAND_PREFIX: &str = "wm.stt.";

/// Incoming topics handled by the daemon.
pub mod incoming {
    /// Speech segment started (from `wm-audio`).
    pub const SPEECH_START: &str = "wm.audio.speech.start";
    /// Speech PCM chunk (from `wm-audio`).
    pub const SPEECH_CHUNK: &str = "wm.audio.speech.chunk";
    /// Speech segment ended (from `wm-audio`).
    pub const SPEECH_END: &str = "wm.audio.speech.end";
    /// Hot-swap the active whisper model (CLI or operator).
    pub const RELOAD_MODEL: &str = "wm.stt.reload_model";
}

/// Outgoing topics published by the daemon.
pub mod outgoing {
    /// Partial in-flight transcript (~500 ms cadence during active speech).
    pub const PARTIAL: &str = "wm.stt.partial";
    /// Finalised transcript with confidence ≥ threshold.
    pub const FINAL: &str = "wm.stt.final";
    /// Finalised transcript with confidence < threshold.
    pub const UNCERTAIN: &str = "wm.stt.uncertain";
    /// Failure marker; payload carries `kind` + `message`.
    pub const ERROR: &str = "wm.stt.error";
    /// Hot-swap completion marker.
    pub const MODEL_LOADED: &str = "wm.stt.model_loaded";
}

/// Decoded request payloads. Returned by [`decode_request`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Request {
    /// `wm.audio.speech.start` payload.
    SpeechStart(SpeechStartEvent),
    /// `wm.audio.speech.chunk` payload.
    SpeechChunk(SpeechChunkEvent),
    /// `wm.audio.speech.end` payload.
    SpeechEnd(SpeechEndEvent),
    /// `wm.stt.reload_model` payload.
    ReloadModel(ReloadModelRequest),
}

/// `wm.audio.speech.start` payload as seen by `wm-stt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechStartEvent {
    /// Unix milliseconds when speech began.
    pub ts: u64,
}

/// `wm.audio.speech.chunk` payload as seen by `wm-stt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechChunkEvent {
    /// Monotonic sequence number within a single utterance.
    pub seq: u64,
    /// Base64-encoded little-endian i16 PCM, 16 kHz mono.
    pub pcm_b64: String,
    /// Unix milliseconds when the chunk was emitted.
    pub ts: u64,
}

/// `wm.audio.speech.end` payload as seen by `wm-stt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechEndEvent {
    /// Duration of the just-completed utterance in milliseconds.
    pub duration_ms: u32,
    /// Unix milliseconds when speech ended.
    pub ts: u64,
}

/// `wm.stt.reload_model` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReloadModelRequest {
    /// New whisper model name (must be in
    /// [`crate::ALLOWED_MODEL_NAMES`]).
    pub model: String,
}

/// Outbound `wm.stt.partial` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartialEvent {
    /// Current best-guess transcript.
    pub text: String,
    /// Unix milliseconds at emission.
    pub ts: u64,
}

/// Outbound `wm.stt.final` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FinalEvent {
    /// Finalised transcript.
    pub text: String,
    /// Confidence in `(0.0, 1.0]`; computed from whisper's
    /// `no_speech_prob` (PRD §2.4).
    pub confidence: f32,
    /// Wall-clock duration of the source audio in milliseconds.
    pub duration_ms: u32,
    /// Whisper model name that produced the transcript.
    pub model: String,
    /// Unix milliseconds at emission.
    pub ts: u64,
}

/// Outbound `wm.stt.uncertain` payload (below confidence threshold).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UncertainEvent {
    /// Low-confidence transcript.
    pub text: String,
    /// Confidence below the active threshold.
    pub confidence: f32,
    /// Unix milliseconds at emission.
    pub ts: u64,
}

/// Outbound `wm.stt.error` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorEvent {
    /// Short kind tag (`"engine" | "model" | "bus" | "io" | "cloud"`).
    pub kind: String,
    /// Human-readable detail.
    pub message: String,
    /// Unix milliseconds at emission.
    pub ts: u64,
}

/// Outbound `wm.stt.model_loaded` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelLoadedEvent {
    /// Newly-active whisper model name.
    pub model: String,
    /// Milliseconds the warmup load took.
    pub warmup_ms: u64,
    /// Unix milliseconds at emission.
    pub ts: u64,
}

/// Errors raised while decoding an inbound payload.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Topic was not one of the known incoming names.
    #[error("unknown topic: {0}")]
    UnknownTopic(String),
    /// JSON decode of the payload failed.
    #[error("payload decode failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// Decode a raw `(topic, data)` pair into a strongly-typed [`Request`].
///
/// # Errors
/// Returns [`DecodeError::UnknownTopic`] for topics outside the
/// known `wm.audio.speech.{start,chunk,end}` and `wm.stt.reload_model`
/// set, or [`DecodeError::Json`] when the payload shape doesn't match.
pub fn decode_request(topic: &str, data: &serde_json::Value) -> Result<Request, DecodeError> {
    match topic {
        incoming::SPEECH_START => Ok(Request::SpeechStart(serde_json::from_value(data.clone())?)),
        incoming::SPEECH_CHUNK => Ok(Request::SpeechChunk(serde_json::from_value(data.clone())?)),
        incoming::SPEECH_END => Ok(Request::SpeechEnd(serde_json::from_value(data.clone())?)),
        incoming::RELOAD_MODEL => {
            Ok(Request::ReloadModel(serde_json::from_value(data.clone())?))
        }
        other => Err(DecodeError::UnknownTopic(other.to_string())),
    }
}

/// Wall-clock milliseconds since the Unix epoch. Saturates to `u64::MAX`
/// if the clock is set before 1970 (shouldn't happen).
#[must_use]
pub fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(u64::MAX, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    reason = "tests"
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decode_speech_start_minimal() {
        let v = json!({ "ts": 1_234_567_890_123_u64 });
        let req = decode_request(incoming::SPEECH_START, &v).expect("decodes");
        match req {
            Request::SpeechStart(s) => assert_eq!(s.ts, 1_234_567_890_123),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn decode_speech_chunk_roundtrip() {
        let v = json!({ "seq": 7, "pcm_b64": "AAAA", "ts": 42_u64 });
        let req = decode_request(incoming::SPEECH_CHUNK, &v).expect("decodes");
        match req {
            Request::SpeechChunk(c) => {
                assert_eq!(c.seq, 7);
                assert_eq!(c.pcm_b64, "AAAA");
                assert_eq!(c.ts, 42);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn decode_speech_end_minimal() {
        let v = json!({ "duration_ms": 5000_u32, "ts": 99_u64 });
        let req = decode_request(incoming::SPEECH_END, &v).expect("decodes");
        match req {
            Request::SpeechEnd(e) => {
                assert_eq!(e.duration_ms, 5000);
                assert_eq!(e.ts, 99);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn decode_reload_model() {
        let v = json!({ "model": "medium.en" });
        let req = decode_request(incoming::RELOAD_MODEL, &v).expect("decodes");
        match req {
            Request::ReloadModel(r) => assert_eq!(r.model, "medium.en"),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn decode_unknown_topic() {
        let v = json!({});
        let err = decode_request("wm.stt.bogus", &v).expect_err("unknown topic rejected");
        assert!(matches!(err, DecodeError::UnknownTopic(t) if t == "wm.stt.bogus"));
    }

    #[test]
    fn decode_speech_start_missing_ts_errors() {
        let v = json!({});
        let err = decode_request(incoming::SPEECH_START, &v).expect_err("missing ts");
        assert!(matches!(err, DecodeError::Json(_)));
    }

    #[test]
    fn decode_speech_chunk_missing_pcm_errors() {
        let v = json!({ "seq": 1, "ts": 1 });
        let err = decode_request(incoming::SPEECH_CHUNK, &v).expect_err("missing pcm");
        assert!(matches!(err, DecodeError::Json(_)));
    }

    #[test]
    fn outbound_final_roundtrip() {
        let f = FinalEvent {
            text: "hello world".to_string(),
            confidence: 0.91,
            duration_ms: 4321,
            model: "distil-small.en".to_string(),
            ts: 17,
        };
        let v = serde_json::to_value(&f).expect("serialises");
        let back: FinalEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back.text, "hello world");
        assert_eq!(back.confidence, 0.91);
        assert_eq!(back.duration_ms, 4321);
        assert_eq!(back.model, "distil-small.en");
        assert_eq!(back.ts, 17);
    }

    #[test]
    fn outbound_uncertain_roundtrip() {
        let u = UncertainEvent {
            text: "mumble".to_string(),
            confidence: 0.2,
            ts: 5,
        };
        let v = serde_json::to_value(&u).expect("serialises");
        let back: UncertainEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back.text, "mumble");
        assert_eq!(back.confidence, 0.2);
        assert_eq!(back.ts, 5);
    }

    #[test]
    fn outbound_error_roundtrip() {
        let e = ErrorEvent {
            kind: "engine".to_string(),
            message: "whisper.cpp returned EIO".to_string(),
            ts: 3,
        };
        let v = serde_json::to_value(&e).expect("serialises");
        let back: ErrorEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back.kind, "engine");
        assert_eq!(back.message, "whisper.cpp returned EIO");
        assert_eq!(back.ts, 3);
    }

    #[test]
    fn outbound_model_loaded_roundtrip() {
        let m = ModelLoadedEvent {
            model: "small.en".to_string(),
            warmup_ms: 1980,
            ts: 100,
        };
        let v = serde_json::to_value(&m).expect("serialises");
        let back: ModelLoadedEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back.model, "small.en");
        assert_eq!(back.warmup_ms, 1980);
        assert_eq!(back.ts, 100);
    }

    #[test]
    fn now_unix_ms_monotonic() {
        let a = now_unix_ms();
        let b = now_unix_ms();
        assert!(b >= a, "clock should not run backwards within a call");
    }

    #[test]
    fn topic_constants_match_strings() {
        assert_eq!(incoming::SPEECH_START, "wm.audio.speech.start");
        assert_eq!(incoming::SPEECH_CHUNK, "wm.audio.speech.chunk");
        assert_eq!(incoming::SPEECH_END, "wm.audio.speech.end");
        assert_eq!(incoming::RELOAD_MODEL, "wm.stt.reload_model");
        assert_eq!(outgoing::PARTIAL, "wm.stt.partial");
        assert_eq!(outgoing::FINAL, "wm.stt.final");
        assert_eq!(outgoing::UNCERTAIN, "wm.stt.uncertain");
        assert_eq!(outgoing::ERROR, "wm.stt.error");
        assert_eq!(outgoing::MODEL_LOADED, "wm.stt.model_loaded");
    }
}
