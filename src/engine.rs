//! Transcription engine abstraction.
//!
//! Iter-4 ships the [`TranscriptionEngine`] trait and a deterministic
//! [`StubEngine`] so the daemon's per-utterance state machine
//! ([`crate::processor`]) can be wired and unit-tested without binding
//! the whisper.cpp / `whisper-rs` C dependency. The real engine lands in
//! iter-5.

use crate::{SttError, validate_model_name};

/// Result of finalising an utterance — the daemon turns this into a
/// [`crate::bus::FinalEvent`] or [`crate::bus::UncertainEvent`] based on
/// the configured confidence threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineFinal {
    /// Finalised transcript text.
    pub text: String,
    /// Confidence in `(0.0, 1.0]`. For the real engine this is
    /// `1.0 - no_speech_prob`; the stub returns a fixed value.
    pub confidence: f32,
}

/// Errors raised by a [`TranscriptionEngine`].
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// A chunk could not be accepted (bad encoding, wrong PCM format, …).
    #[error("engine rejected chunk: {reason}")]
    BadChunk {
        /// Human-readable detail.
        reason: String,
    },
    /// The active model is not on the allow-list.
    #[error(transparent)]
    UnknownModel(#[from] SttError),
    /// Engine internal failure (whisper.cpp init, decoder crash, …).
    #[error("engine internal failure: {0}")]
    Internal(String),
}

/// Streaming transcription contract.
///
/// Lifecycle (one utterance):
/// 1. [`Self::reset`] before the first chunk of a new utterance.
/// 2. Zero or more [`Self::accept_chunk`] calls.
/// 3. [`Self::current_partial`] queried in between for partial emissions.
/// 4. [`Self::finalise`] once `speech.end` arrives.
///
/// [`Self::reload_model`] may be called between utterances (the daemon
/// drains the in-flight one first).
pub trait TranscriptionEngine: Send {
    /// Currently active model name (must be in [`crate::ALLOWED_MODEL_NAMES`]).
    fn model_name(&self) -> &str;

    /// Feed a chunk of PCM (base64 little-endian i16 @ 16 kHz mono).
    ///
    /// # Errors
    /// Returns [`EngineError::BadChunk`] for unparseable or wrong-shape
    /// input; [`EngineError::Internal`] for decoder failures.
    fn accept_chunk(&mut self, seq: u64, pcm_b64: &str) -> Result<(), EngineError>;

    /// Best-guess transcript so far without finalising. `None` before any
    /// chunk has been accepted in the current utterance.
    fn current_partial(&mut self) -> Option<String>;

    /// Finalise the active utterance.
    ///
    /// # Errors
    /// Returns [`EngineError::Internal`] if the engine cannot produce a
    /// final transcript.
    fn finalise(&mut self, duration_ms: u32) -> Result<EngineFinal, EngineError>;

    /// Hot-swap the active model. Returns warmup milliseconds the swap
    /// took (PRD §2.3 — ~2 s on the real engine, 0 on the stub).
    ///
    /// # Errors
    /// Returns [`EngineError::UnknownModel`] if `name` is not in
    /// [`crate::ALLOWED_MODEL_NAMES`]. Real engine may also surface
    /// [`EngineError::Internal`] on `.bin` load failure.
    fn reload_model(&mut self, name: &str) -> Result<u64, EngineError>;

    /// Drop all per-utterance state. Called by the daemon between
    /// utterances.
    fn reset(&mut self);
}

/// Deterministic stub engine — no whisper.cpp. Used by tests and as the
/// daemon's default until iter-5 lands the real `whisper-rs` binding.
#[derive(Debug, Clone)]
pub struct StubEngine {
    model: String,
    chunks: u64,
    fixed_confidence: f32,
}

impl StubEngine {
    /// Construct a stub with the given model name (must validate) and a
    /// fixed confidence value the finaliser will report.
    ///
    /// # Errors
    /// Returns [`SttError::UnknownModel`] if `model` is not in the
    /// allow-list, or [`SttError::InvalidThreshold`] when
    /// `fixed_confidence` is outside `(0.0, 1.0]`.
    pub fn new(model: impl Into<String>, fixed_confidence: f32) -> Result<Self, SttError> {
        let model = model.into();
        validate_model_name(&model)?;
        crate::validate_threshold(fixed_confidence)?;
        Ok(Self {
            model,
            chunks: 0,
            fixed_confidence,
        })
    }

    /// Convenience: stub at the project default model + threshold.
    ///
    /// # Errors
    /// Same as [`Self::new`]; only triggers if the project defaults are
    /// changed to an invalid combination.
    pub fn default_for_tests() -> Result<Self, SttError> {
        Self::new(crate::DEFAULT_MODEL_NAME, 0.75)
    }
}

impl TranscriptionEngine for StubEngine {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn accept_chunk(&mut self, _seq: u64, pcm_b64: &str) -> Result<(), EngineError> {
        if pcm_b64.is_empty() {
            return Err(EngineError::BadChunk {
                reason: "empty pcm_b64".to_string(),
            });
        }
        self.chunks = self.chunks.saturating_add(1);
        Ok(())
    }

    fn current_partial(&mut self) -> Option<String> {
        if self.chunks == 0 {
            None
        } else {
            Some(format!("[stub partial {}]", self.chunks))
        }
    }

    fn finalise(&mut self, duration_ms: u32) -> Result<EngineFinal, EngineError> {
        Ok(EngineFinal {
            text: format!("[stub final {duration_ms}ms]"),
            confidence: self.fixed_confidence,
        })
    }

    fn reload_model(&mut self, name: &str) -> Result<u64, EngineError> {
        validate_model_name(name)?;
        self.model = name.to_string();
        Ok(0)
    }

    fn reset(&mut self) {
        self.chunks = 0;
    }
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

    #[test]
    fn stub_constructs_with_defaults() {
        let e = StubEngine::default_for_tests().expect("stub builds");
        assert_eq!(e.model_name(), crate::DEFAULT_MODEL_NAME);
    }

    #[test]
    fn stub_rejects_unknown_model() {
        let err = StubEngine::new("tiny.en", 0.5).expect_err("unknown model");
        assert!(matches!(err, SttError::UnknownModel { .. }));
    }

    #[test]
    fn stub_rejects_bad_threshold() {
        let err = StubEngine::new("small.en", 0.0).expect_err("threshold out of range");
        assert!(matches!(err, SttError::InvalidThreshold(_)));
    }

    #[test]
    fn partial_is_none_before_any_chunk() {
        let mut e = StubEngine::default_for_tests().unwrap();
        assert!(e.current_partial().is_none());
    }

    #[test]
    fn accept_chunk_advances_partial() {
        let mut e = StubEngine::default_for_tests().unwrap();
        e.accept_chunk(0, "AAAA").unwrap();
        assert_eq!(e.current_partial().as_deref(), Some("[stub partial 1]"));
        e.accept_chunk(1, "BBBB").unwrap();
        assert_eq!(e.current_partial().as_deref(), Some("[stub partial 2]"));
    }

    #[test]
    fn accept_chunk_rejects_empty_pcm() {
        let mut e = StubEngine::default_for_tests().unwrap();
        let err = e.accept_chunk(0, "").expect_err("empty pcm rejected");
        assert!(matches!(err, EngineError::BadChunk { .. }));
    }

    #[test]
    fn finalise_uses_fixed_confidence() {
        let mut e = StubEngine::new("medium.en", 0.42).unwrap();
        e.accept_chunk(0, "AAAA").unwrap();
        let f = e.finalise(1234).unwrap();
        assert_eq!(f.text, "[stub final 1234ms]");
        assert_eq!(f.confidence, 0.42);
    }

    #[test]
    fn reset_clears_chunks() {
        let mut e = StubEngine::default_for_tests().unwrap();
        e.accept_chunk(0, "AAAA").unwrap();
        e.reset();
        assert!(e.current_partial().is_none());
    }

    #[test]
    fn reload_model_swaps_name() {
        let mut e = StubEngine::default_for_tests().unwrap();
        let warmup = e.reload_model("large-v3-turbo").unwrap();
        assert_eq!(warmup, 0);
        assert_eq!(e.model_name(), "large-v3-turbo");
    }

    #[test]
    fn reload_model_rejects_unknown() {
        let mut e = StubEngine::default_for_tests().unwrap();
        let err = e.reload_model("tiny.en").expect_err("unknown rejected");
        assert!(matches!(err, EngineError::UnknownModel(SttError::UnknownModel { .. })));
        assert_eq!(e.model_name(), crate::DEFAULT_MODEL_NAME);
    }
}
