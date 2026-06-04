//! Per-utterance state machine for the `wm-stt` daemon.
//!
//! [`UtteranceProcessor`] folds a stream of [`Request`] inputs (decoded
//! from agorabus via [`crate::bus::decode_request`]) into [`Emit`]
//! outputs the daemon should publish. Pure-data, no I/O — the live
//! tokio subscribe loop and publisher (iter-5) drives it.
//!
//! State machine:
//!
//! ```text
//!         SpeechStart                SpeechEnd
//!   Idle ─────────────►  Speaking ─────────────► Idle
//!                       │   ▲
//!                       │   │ SpeechChunk (throttled partial emit)
//!                       └───┘
//! ```
//!
//! `ReloadModel` is honoured immediately when [`State::Idle`]; while
//! [`State::Speaking`] it is queued and applied right after the active
//! utterance finalises (PRD §2.3: "Daemon completes in-flight
//! transcription before swap").

use crate::bus::{
    ErrorEvent, FinalEvent, ModelLoadedEvent, PartialEvent, Request, UncertainEvent,
};
use crate::engine::{EngineError, TranscriptionEngine};
use crate::{SttConfig, validate_model_name};

/// Partial-emit cadence in milliseconds. PRD §2.1.5 — "~500 ms".
pub const DEFAULT_PARTIAL_CADENCE_MS: u64 = 500;

/// Minimum speech window duration (ms) before we run inference.
/// Windows shorter than this are likely false-positive wakes.
pub const MIN_WINDOW_MS: u32 = 200;

/// Maximum speech window duration (ms) before we skip inference.
/// Windows longer than this indicate the end-of-speech detector is stuck.
pub const MAX_WINDOW_MS: u32 = 30_000;

/// Output of [`UtteranceProcessor::handle`]. The live loop publishes
/// each variant on the matching `wm.stt.*` topic.
#[derive(Debug, Clone, PartialEq)]
pub enum Emit {
    /// `wm.stt.partial`
    Partial(PartialEvent),
    /// `wm.stt.final`
    Final(FinalEvent),
    /// `wm.stt.uncertain`
    Uncertain(UncertainEvent),
    /// `wm.stt.error`
    Error(ErrorEvent),
    /// `wm.stt.model_loaded`
    ModelLoaded(ModelLoadedEvent),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Idle,
    Speaking {
        start_ts: u64,
        last_partial_ts: u64,
    },
}

/// Per-utterance state machine. Owns the [`TranscriptionEngine`] and
/// emits events as inputs arrive.
#[derive(Debug)]
pub struct UtteranceProcessor<E: TranscriptionEngine> {
    engine: E,
    config: SttConfig,
    state: State,
    partial_cadence_ms: u64,
    pending_reload: Option<String>,
}

impl<E: TranscriptionEngine> UtteranceProcessor<E> {
    /// Build a processor from an engine plus the active [`SttConfig`].
    /// Uses [`DEFAULT_PARTIAL_CADENCE_MS`] for the partial throttle.
    #[must_use]
    pub const fn new(engine: E, config: SttConfig) -> Self {
        Self {
            engine,
            config,
            state: State::Idle,
            partial_cadence_ms: DEFAULT_PARTIAL_CADENCE_MS,
            pending_reload: None,
        }
    }

    /// Override the partial-emit cadence. Useful for tests.
    #[must_use]
    pub const fn with_partial_cadence_ms(mut self, ms: u64) -> Self {
        self.partial_cadence_ms = ms;
        self
    }

    /// Active engine model name. Exposed for diagnostics.
    pub fn model_name(&self) -> &str {
        self.engine.model_name()
    }

    /// Currently configured confidence threshold.
    #[must_use]
    pub const fn confidence_threshold(&self) -> f32 {
        self.config.confidence_threshold
    }

    /// True when the processor is between utterances.
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        matches!(self.state, State::Idle)
    }

    /// Fold a single decoded request into zero or more emissions.
    ///
    /// `now_ms` is wall-clock at the moment the daemon dequeues the
    /// request; tests inject a deterministic clock so the partial
    /// throttle is reproducible.
    pub fn handle(&mut self, req: Request, now_ms: u64) -> Vec<Emit> {
        match req {
            Request::SpeechStart(s) => self.on_speech_start(s.ts, now_ms),
            Request::SpeechChunk(c) => self.on_speech_chunk(c.seq, &c.pcm_b64, c.ts),
            Request::SpeechEnd(e) => self.on_speech_end(e.duration_ms, e.ts, now_ms),
            Request::ReloadModel(r) => self.on_reload_model(&r.model, now_ms),
        }
    }

    fn on_speech_start(&mut self, start_ts: u64, now_ms: u64) -> Vec<Emit> {
        let mut out = Vec::new();
        if matches!(self.state, State::Speaking { .. }) {
            out.push(Emit::Error(ErrorEvent {
                kind: "protocol".to_string(),
                message: "speech.start while already speaking; dropping in-flight utterance"
                    .to_string(),
                ts: now_ms,
            }));
            self.engine.reset();
        }
        self.state = State::Speaking {
            start_ts,
            last_partial_ts: start_ts,
        };
        out
    }

    fn on_speech_chunk(&mut self, seq: u64, pcm_b64: &str, chunk_ts: u64) -> Vec<Emit> {
        let mut out = Vec::new();
        let State::Speaking {
            start_ts: _,
            last_partial_ts,
        } = self.state
        else {
            out.push(Emit::Error(ErrorEvent {
                kind: "protocol".to_string(),
                message: "speech.chunk while idle; ignoring".to_string(),
                ts: chunk_ts,
            }));
            return out;
        };

        if let Err(err) = self.engine.accept_chunk(seq, pcm_b64) {
            out.push(Emit::Error(engine_error_event(&err, chunk_ts)));
            return out;
        }

        if chunk_ts.saturating_sub(last_partial_ts) >= self.partial_cadence_ms {
            if let Some(text) = self.engine.current_partial_fast() {
                out.push(Emit::Partial(PartialEvent {
                    text,
                    ts: chunk_ts,
                }));
            }
            if let State::Speaking {
                ref mut last_partial_ts,
                ..
            } = self.state
            {
                *last_partial_ts = chunk_ts;
            }
        }
        out
    }

    fn on_speech_end(&mut self, duration_ms: u32, end_ts: u64, now_ms: u64) -> Vec<Emit> {
        let mut out = Vec::new();
        if !matches!(self.state, State::Speaking { .. }) {
            out.push(Emit::Error(ErrorEvent {
                kind: "protocol".to_string(),
                message: "speech.end while idle; ignoring".to_string(),
                ts: end_ts,
            }));
            return out;
        }
        // Window validation: skip inference for invalid window lengths.
        // A < 200 ms window is a VAD false-positive (wake-word tail, click,
        // brief noise) — drop it SILENTLY. Emitting Uncertain here makes
        // wm-dialog ask "could you repeat that?" and derails the real command
        // that follows the blip. A > 30 s window is a stuck detector worth
        // surfacing, so that case still emits Uncertain.
        if duration_ms < MIN_WINDOW_MS {
            self.engine.reset();
            self.state = State::Idle;
            if let Some(pending) = self.pending_reload.take() {
                out.extend(self.apply_reload(&pending, now_ms));
            }
            return out;
        }
        if duration_ms > MAX_WINDOW_MS {
            self.engine.reset();
            self.state = State::Idle;
            out.push(Emit::Uncertain(UncertainEvent {
                text: String::new(),
                confidence: 0.0,
                reason: Some("window_too_long".to_string()),
                ts: end_ts,
            }));
            if let Some(pending) = self.pending_reload.take() {
                out.extend(self.apply_reload(&pending, now_ms));
            }
            return out;
        }
        match self.engine.finalise(duration_ms) {
            Ok(f) => {
                let model = self.engine.model_name().to_string();
                if f.confidence >= self.config.confidence_threshold {
                    out.push(Emit::Final(FinalEvent {
                        text: f.text,
                        confidence: f.confidence,
                        duration_ms,
                        model,
                        ts: end_ts,
                    }));
                } else {
                    out.push(Emit::Uncertain(UncertainEvent {
                        text: f.text,
                        confidence: f.confidence,
                        reason: None,
                        ts: end_ts,
                    }));
                }
            }
            Err(err) => {
                out.push(Emit::Error(engine_error_event(&err, end_ts)));
            }
        }
        self.engine.reset();
        self.state = State::Idle;
        if let Some(pending) = self.pending_reload.take() {
            out.extend(self.apply_reload(&pending, now_ms));
        }
        out
    }

    fn on_reload_model(&mut self, name: &str, now_ms: u64) -> Vec<Emit> {
        if let Err(err) = validate_model_name(name) {
            return vec![Emit::Error(ErrorEvent {
                kind: "model".to_string(),
                message: err.to_string(),
                ts: now_ms,
            })];
        }
        if matches!(self.state, State::Speaking { .. }) {
            self.pending_reload = Some(name.to_string());
            return Vec::new();
        }
        self.apply_reload(name, now_ms)
    }

    fn apply_reload(&mut self, name: &str, now_ms: u64) -> Vec<Emit> {
        match self.engine.reload_model(name) {
            Ok(warmup_ms) => vec![Emit::ModelLoaded(ModelLoadedEvent {
                model: name.to_string(),
                warmup_ms,
                ts: now_ms,
            })],
            Err(err) => vec![Emit::Error(ErrorEvent {
                kind: "model".to_string(),
                message: err.to_string(),
                ts: now_ms,
            })],
        }
    }

    /// Borrow the engine. Diagnostics and tests only.
    pub const fn engine(&self) -> &E {
        &self.engine
    }
}

fn engine_error_event(err: &EngineError, ts: u64) -> ErrorEvent {
    engine_error_event_pub(err, ts)
}

/// Exposed for tests in sibling modules (`whisper_engine::tests`).
#[doc(hidden)]
pub(crate) fn engine_error_event_pub(err: &EngineError, ts: u64) -> ErrorEvent {
    let kind = match err {
        EngineError::BadChunk { .. } => "io",
        EngineError::UnknownModel(_) => "model",
        EngineError::ModelMissing { .. } => "model_missing",
        EngineError::Internal(_) => "engine",
    };
    ErrorEvent {
        kind: kind.to_string(),
        message: err.to_string(),
        ts,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use super::*;
    use crate::bus::{ReloadModelRequest, SpeechChunkEvent, SpeechEndEvent, SpeechStartEvent};
    use crate::engine::StubEngine;

    fn make(threshold: f32, fixed_confidence: f32) -> UtteranceProcessor<StubEngine> {
        let engine = StubEngine::new("distil-small.en", fixed_confidence).unwrap();
        let cfg = SttConfig {
            confidence_threshold: threshold,
            ..SttConfig::default()
        };
        UtteranceProcessor::new(engine, cfg)
    }

    #[test]
    fn idle_is_initial_state() {
        let p = make(0.45, 0.9);
        assert!(p.is_idle());
        assert_eq!(p.model_name(), "distil-small.en");
    }

    #[test]
    fn speech_start_then_end_emits_final_when_confidence_high() {
        let mut p = make(0.45, 0.8);
        let s = p.handle(Request::SpeechStart(SpeechStartEvent { ts: 100 }), 100);
        assert!(s.is_empty(), "start emits nothing");
        let e = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 5000,
                ts: 5100,
            }),
            5100,
        );
        assert_eq!(e.len(), 1);
        match &e[0] {
            Emit::Final(f) => {
                assert_eq!(f.duration_ms, 5000);
                assert_eq!(f.confidence, 0.8);
                assert_eq!(f.model, "distil-small.en");
            }
            other => panic!("expected Final, got {other:?}"),
        }
        assert!(p.is_idle());
    }

    #[test]
    fn low_confidence_emits_uncertain() {
        let mut p = make(0.45, 0.2);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let e = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 1000,
                ts: 1000,
            }),
            1000,
        );
        match &e[0] {
            Emit::Uncertain(u) => assert_eq!(u.confidence, 0.2),
            other => panic!("expected Uncertain, got {other:?}"),
        }
    }

    #[test]
    fn chunks_emit_partials_at_cadence() {
        let mut p = make(0.45, 0.9).with_partial_cadence_ms(500);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);

        let r1 = p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 0,
                pcm_b64: "AAAA".to_string(),
                ts: 200,
            }),
            200,
        );
        assert!(r1.is_empty(), "200 ms < 500 ms cadence: no partial");

        let r2 = p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 1,
                pcm_b64: "BBBB".to_string(),
                ts: 600,
            }),
            600,
        );
        assert_eq!(r2.len(), 1, "600 ms >= 500 ms cadence: partial emits");
        assert!(matches!(r2[0], Emit::Partial(_)));

        let r3 = p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 2,
                pcm_b64: "CCCC".to_string(),
                ts: 900,
            }),
            900,
        );
        assert!(r3.is_empty(), "900-600 = 300 ms < cadence: no partial");

        let r4 = p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 3,
                pcm_b64: "DDDD".to_string(),
                ts: 1200,
            }),
            1200,
        );
        assert_eq!(r4.len(), 1, "1200-600 = 600 ms >= cadence: partial");
    }

    #[test]
    fn chunk_while_idle_emits_protocol_error() {
        let mut p = make(0.45, 0.9);
        let r = p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 0,
                pcm_b64: "AAAA".to_string(),
                ts: 1,
            }),
            1,
        );
        match &r[0] {
            Emit::Error(e) => assert_eq!(e.kind, "protocol"),
            other => panic!("expected protocol Error, got {other:?}"),
        }
    }

    #[test]
    fn end_while_idle_emits_protocol_error() {
        let mut p = make(0.45, 0.9);
        let r = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 1,
                ts: 1,
            }),
            1,
        );
        match &r[0] {
            Emit::Error(e) => assert_eq!(e.kind, "protocol"),
            other => panic!("expected protocol Error, got {other:?}"),
        }
        assert!(p.is_idle());
    }

    #[test]
    fn double_start_resets_and_errors() {
        let mut p = make(0.45, 0.9).with_partial_cadence_ms(100);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 0,
                pcm_b64: "AAAA".to_string(),
                ts: 150,
            }),
            150,
        );
        let r = p.handle(Request::SpeechStart(SpeechStartEvent { ts: 200 }), 200);
        assert_eq!(r.len(), 1);
        match &r[0] {
            Emit::Error(e) => assert_eq!(e.kind, "protocol"),
            other => panic!("expected protocol Error, got {other:?}"),
        }
    }

    #[test]
    fn bad_chunk_emits_io_error() {
        let mut p = make(0.45, 0.9);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r = p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 0,
                pcm_b64: String::new(),
                ts: 100,
            }),
            100,
        );
        match &r[0] {
            Emit::Error(e) => {
                assert_eq!(e.kind, "io");
                assert!(e.message.contains("empty pcm_b64"));
            }
            other => panic!("expected io Error, got {other:?}"),
        }
    }

    #[test]
    fn reload_while_idle_emits_model_loaded() {
        let mut p = make(0.45, 0.9);
        let r = p.handle(
            Request::ReloadModel(ReloadModelRequest {
                model: "small.en".to_string(),
            }),
            42,
        );
        assert_eq!(r.len(), 1);
        match &r[0] {
            Emit::ModelLoaded(m) => {
                assert_eq!(m.model, "small.en");
                assert_eq!(m.ts, 42);
            }
            other => panic!("expected ModelLoaded, got {other:?}"),
        }
        assert_eq!(p.model_name(), "small.en");
    }

    #[test]
    fn reload_during_speech_is_queued_until_end() {
        let mut p = make(0.45, 0.9);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r1 = p.handle(
            Request::ReloadModel(ReloadModelRequest {
                model: "medium.en".to_string(),
            }),
            500,
        );
        assert!(r1.is_empty(), "reload during speech defers silently");
        assert_eq!(
            p.model_name(),
            "distil-small.en",
            "model unchanged while speaking"
        );

        let r2 = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 1000,
                ts: 1000,
            }),
            1000,
        );
        let kinds: Vec<&str> = r2
            .iter()
            .map(|e| match e {
                Emit::Final(_) => "final",
                Emit::Uncertain(_) => "uncertain",
                Emit::Partial(_) => "partial",
                Emit::Error(_) => "error",
                Emit::ModelLoaded(_) => "model_loaded",
            })
            .collect();
        assert_eq!(kinds, vec!["final", "model_loaded"]);
        assert_eq!(p.model_name(), "medium.en");
    }

    #[test]
    fn reload_unknown_model_emits_error() {
        let mut p = make(0.45, 0.9);
        let r = p.handle(
            Request::ReloadModel(ReloadModelRequest {
                model: "tiny.en".to_string(),
            }),
            10,
        );
        match &r[0] {
            Emit::Error(e) => assert_eq!(e.kind, "model"),
            other => panic!("expected model Error, got {other:?}"),
        }
    }

    #[test]
    fn threshold_exact_equal_is_final() {
        let mut p = make(0.5, 0.5);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        // Use MIN_WINDOW_MS (200 ms) so the window-validation gate passes.
        let r = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: MIN_WINDOW_MS,
                ts: u64::from(MIN_WINDOW_MS),
            }),
            u64::from(MIN_WINDOW_MS),
        );
        assert!(matches!(r[0], Emit::Final(_)));
    }

    /// Engine that records which partial path was called. Lets the
    /// processor test prove the cadence emit goes through the
    /// `current_partial_fast` (cheap) path, not `current_partial`.
    struct PartialPathSpy {
        model: String,
        fast_calls: std::cell::Cell<u32>,
        slow_calls: std::cell::Cell<u32>,
    }

    impl crate::engine::TranscriptionEngine for PartialPathSpy {
        fn model_name(&self) -> &str {
            &self.model
        }
        fn accept_chunk(
            &mut self,
            _seq: u64,
            _pcm_b64: &str,
        ) -> Result<(), crate::engine::EngineError> {
            Ok(())
        }
        fn current_partial(&mut self) -> Option<String> {
            self.slow_calls.set(self.slow_calls.get() + 1);
            Some("SLOW".to_string())
        }
        fn current_partial_fast(&mut self) -> Option<String> {
            self.fast_calls.set(self.fast_calls.get() + 1);
            Some("FAST".to_string())
        }
        fn finalise(
            &mut self,
            _duration_ms: u32,
        ) -> Result<crate::engine::EngineFinal, crate::engine::EngineError> {
            Ok(crate::engine::EngineFinal {
                text: String::new(),
                confidence: 1.0,
            })
        }
        fn reload_model(&mut self, _name: &str) -> Result<u64, crate::engine::EngineError> {
            Ok(0)
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn partial_cadence_uses_fast_path() {
        let engine = PartialPathSpy {
            model: "distil-small.en".to_string(),
            fast_calls: std::cell::Cell::new(0),
            slow_calls: std::cell::Cell::new(0),
        };
        let cfg = SttConfig {
            confidence_threshold: 0.5,
            ..SttConfig::default()
        };
        let mut p = UtteranceProcessor::new(engine, cfg).with_partial_cadence_ms(500);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r = p.handle(
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 0,
                pcm_b64: "AAAA".to_string(),
                ts: 600,
            }),
            600,
        );
        match &r[0] {
            Emit::Partial(pe) => assert_eq!(pe.text, "FAST"),
            other => panic!("expected Partial, got {other:?}"),
        }
        assert_eq!(p.engine.fast_calls.get(), 1);
        assert_eq!(p.engine.slow_calls.get(), 0);
    }

    // --- AC5: window validation ---

    /// Zero-duration window (speech.start immediately followed by speech.end
    /// with duration_ms = 0) must publish `wm.stt.uncertain` with
    /// `reason = "window_invalid"` and must not crash.
    #[test]
    fn empty_window_emits_uncertain_window_invalid() {
        let mut p = make(0.45, 0.9);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 0,
                ts: 0,
            }),
            0,
        );
        assert_eq!(r.len(), 1, "exactly one emit for zero-length window");
        match &r[0] {
            Emit::Uncertain(u) => {
                assert_eq!(u.reason.as_deref(), Some("window_invalid"));
                assert_eq!(u.confidence, 0.0);
            }
            other => panic!("expected window_invalid Uncertain, got {other:?}"),
        }
        assert!(p.is_idle(), "processor returns to idle after window_invalid");
    }

    /// Window just below MIN_WINDOW_MS threshold must be rejected.
    #[test]
    fn window_below_min_emits_uncertain_window_invalid() {
        let mut p = make(0.45, 0.9);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: MIN_WINDOW_MS - 1,
                ts: 199,
            }),
            199,
        );
        assert_eq!(r.len(), 1);
        match &r[0] {
            Emit::Uncertain(u) => {
                assert_eq!(u.reason.as_deref(), Some("window_invalid"));
            }
            other => panic!("expected window_invalid, got {other:?}"),
        }
    }

    /// Window just above MAX_WINDOW_MS threshold must be rejected.
    #[test]
    fn window_above_max_emits_uncertain_window_invalid() {
        let mut p = make(0.45, 0.9);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: MAX_WINDOW_MS + 1,
                ts: u64::from(MAX_WINDOW_MS) + 1,
            }),
            u64::from(MAX_WINDOW_MS) + 1,
        );
        assert_eq!(r.len(), 1);
        match &r[0] {
            Emit::Uncertain(u) => {
                assert_eq!(u.reason.as_deref(), Some("window_invalid"));
            }
            other => panic!("expected window_invalid, got {other:?}"),
        }
    }

    /// Valid window length at the boundary does NOT emit window_invalid.
    #[test]
    fn valid_window_at_min_boundary_is_not_rejected() {
        let mut p = make(0.45, 0.9);
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: MIN_WINDOW_MS,
                ts: u64::from(MIN_WINDOW_MS),
            }),
            u64::from(MIN_WINDOW_MS),
        );
        assert_eq!(r.len(), 1);
        // Should be Final or Uncertain (confidence based), but NOT window_invalid.
        match &r[0] {
            Emit::Uncertain(u) => {
                assert_ne!(u.reason.as_deref(), Some("window_invalid"),
                    "min-boundary window must not be rejected as window_invalid");
            }
            Emit::Final(_) => {} // also acceptable
            other => panic!("expected Final or Uncertain, got {other:?}"),
        }
    }

    // --- AC9: confidence threshold ---

    /// Normal uncertain (below threshold) has no reason field.
    #[test]
    fn low_confidence_uncertain_has_no_window_invalid_reason() {
        let mut p = make(0.45, 0.2); // confidence 0.2 < threshold 0.45
        p.handle(Request::SpeechStart(SpeechStartEvent { ts: 0 }), 0);
        let r = p.handle(
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: MIN_WINDOW_MS,
                ts: u64::from(MIN_WINDOW_MS),
            }),
            u64::from(MIN_WINDOW_MS),
        );
        match &r[0] {
            Emit::Uncertain(u) => {
                assert!(
                    u.reason.is_none(),
                    "low-confidence uncertain must not set reason"
                );
                assert_eq!(u.confidence, 0.2);
            }
            other => panic!("expected Uncertain, got {other:?}"),
        }
    }
}
