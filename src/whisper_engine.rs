//! `whisper-rs` engine binding for `wm-stt` (PRD §1, §2.1, §2.3).
//!
//! Enabled via the `whisper` cargo feature; the default build keeps this
//! module out of the binary so iterating on the daemon doesn't require
//! the whisper.cpp toolchain. Activation requires a `.bin` model file
//! at `<models_root>/whisper-<sanitised-name>.bin` — for example
//! `/usr/share/wintermute/models/whisper-distil-small-en.bin` for the
//! default `distil-small.en` model.
//!
//! iter-6 ships a complete [`WhisperEngine`] impl of
//! [`TranscriptionEngine`]: model load on construction, base64 PCM
//! accumulation in [`Self::accept_chunk`], full inference on
//! [`Self::finalise`] (single pass, no streaming partials yet), hot
//! swap via [`Self::reload_model`]. iter-7 will hoist `finalise` into
//! `tokio::task::spawn_blocking` and add cheap mid-stream partials.

#![cfg(feature = "whisper")]

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use base64::Engine as _;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::engine::{EngineError, EngineFinal, TranscriptionEngine};
use crate::validate_model_name;

/// Production whisper.cpp engine for `wm-stt`.
///
/// Holds one [`WhisperContext`] per active model — [`Self::reload_model`]
/// swaps it in place under the internal mutex. PCM chunks accumulate in
/// [`Self::buffer`] as `f32` samples normalised to `[-1.0, 1.0]`;
/// [`Self::finalise`] hands the buffer to whisper.cpp.
pub struct WhisperEngine {
    ctx: Mutex<WhisperContext>,
    model_name: String,
    models_root: PathBuf,
    buffer: Vec<f32>,
}

impl std::fmt::Debug for WhisperEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperEngine")
            .field("model_name", &self.model_name)
            .field("models_root", &self.models_root)
            .field("buffer_samples", &self.buffer.len())
            .finish()
    }
}

impl WhisperEngine {
    /// Load `model_name` from `models_root/whisper-<name>.bin`.
    ///
    /// # Errors
    /// - [`EngineError::UnknownModel`] when `model_name` is not in
    ///   [`crate::ALLOWED_MODEL_NAMES`].
    /// - [`EngineError::Internal`] when whisper.cpp rejects the `.bin`
    ///   file (missing, corrupt, wrong arch) or when the resolved path
    ///   contains non-UTF-8 bytes.
    pub fn load(
        model_name: &str,
        models_root: impl Into<PathBuf>,
    ) -> Result<Self, EngineError> {
        validate_model_name(model_name).map_err(EngineError::from)?;
        let root: PathBuf = models_root.into();
        let path = model_file_path(&root, model_name);
        let ctx = open_context(&path)?;
        Ok(Self {
            ctx: Mutex::new(ctx),
            model_name: model_name.to_string(),
            models_root: root,
            buffer: Vec::new(),
        })
    }

    /// Number of `f32` samples buffered for the active utterance.
    /// Exposed for diagnostics and tests; not part of the trait.
    #[must_use]
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }
}

/// Resolve `<root>/whisper-<sanitised-name>.bin`. Dots in the model name
/// become hyphens so the on-disk file is shell-friendly: the default
/// `distil-small.en` maps to `whisper-distil-small-en.bin`.
fn model_file_path(root: &Path, name: &str) -> PathBuf {
    root.join(format!("whisper-{}.bin", name.replace('.', "-")))
}

fn open_context(path: &Path) -> Result<WhisperContext, EngineError> {
    // Fast path: check for file existence before invoking whisper.cpp so we
    // can return `ModelMissing` (maps to `wm.stt.error { kind: "model_missing" }`)
    // instead of a generic `Internal` error that masks the root cause.
    if !path.exists() {
        return Err(EngineError::ModelMissing {
            path: path.display().to_string(),
        });
    }
    let params = WhisperContextParameters::default();
    let path_str = path.to_str().ok_or_else(|| {
        EngineError::Internal(format!("non-utf8 model path: {}", path.display()))
    })?;
    WhisperContext::new_with_params(path_str, params)
        .map_err(|e| EngineError::Internal(format!("whisper load {path_str}: {e}")))
}

/// Decode a base64-encoded little-endian i16 PCM chunk into the f32
/// samples whisper.cpp expects. Pure function; exposed for unit-test
/// reach.
///
/// # Errors
/// Returns [`EngineError::BadChunk`] for empty input, base64 decode
/// failures, or an odd byte count (i16 PCM must have an even length).
pub fn decode_pcm_b64(pcm_b64: &str) -> Result<Vec<f32>, EngineError> {
    if pcm_b64.is_empty() {
        return Err(EngineError::BadChunk {
            reason: "empty pcm_b64".to_string(),
        });
    }
    let raw = base64::engine::general_purpose::STANDARD
        .decode(pcm_b64)
        .map_err(|e| EngineError::BadChunk {
            reason: format!("base64 decode: {e}"),
        })?;
    if !raw.len().is_multiple_of(2) {
        return Err(EngineError::BadChunk {
            reason: format!("odd byte count {}; i16 PCM expected", raw.len()),
        });
    }
    let mut out = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        let Ok(bytes): Result<[u8; 2], _> = chunk.try_into() else {
            continue;
        };
        let sample = i16::from_le_bytes(bytes);
        // i16::MAX normalised to 1.0; i16::MIN to ~-1.0. as_conversions
        // is the canonical way to express f32 from i16 — clippy lints
        // are warnings, not errors, in the feature-gated path.
        #[allow(clippy::as_conversions)]
        let normalised = sample as f32 / f32::from(i16::MAX);
        out.push(normalised);
    }
    Ok(out)
}

impl TranscriptionEngine for WhisperEngine {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn accept_chunk(&mut self, _seq: u64, pcm_b64: &str) -> Result<(), EngineError> {
        let samples = decode_pcm_b64(pcm_b64)?;
        self.buffer.extend(samples);
        Ok(())
    }

    fn current_partial(&mut self) -> Option<String> {
        // iter-6: no mid-stream partial decoding — whisper.cpp's
        // streaming hooks (`encode` / `decode` split) land in iter-7
        // alongside `spawn_blocking` so the partial path doesn't
        // serialise behind the final inference.
        None
    }

    fn finalise(&mut self, _duration_ms: u32) -> Result<EngineFinal, EngineError> {
        let samples = std::mem::take(&mut self.buffer);
        if samples.is_empty() {
            return Ok(EngineFinal {
                text: String::new(),
                confidence: 0.0,
            });
        }
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(num_threads());
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        let ctx = self
            .ctx
            .lock()
            .map_err(|e| EngineError::Internal(format!("ctx poisoned: {e}")))?;
        let mut state = ctx
            .create_state()
            .map_err(|e| EngineError::Internal(format!("create_state: {e}")))?;
        state
            .full(params, &samples)
            .map_err(|e| EngineError::Internal(format!("full inference: {e}")))?;
        let n_segments = state
            .full_n_segments()
            .map_err(|e| EngineError::Internal(format!("n_segments: {e}")))?;
        let mut text = String::new();
        // Confidence is the average per-token probability across all segments.
        // whisper-rs 0.13 does not expose per-segment no_speech_prob; the
        // token-level `full_get_token_prob` is the best available signal.
        let mut token_prob_sum = 0.0_f32;
        let mut token_count = 0_i32;
        for i in 0..n_segments {
            let seg_text = state
                .full_get_segment_text(i)
                .map_err(|e| EngineError::Internal(format!("seg text {i}: {e}")))?;
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(seg_text.trim());
            let n_tokens = state
                .full_n_tokens(i)
                .map_err(|e| EngineError::Internal(format!("n_tokens seg {i}: {e}")))?;
            for t in 0..n_tokens {
                let p = state
                    .full_get_token_prob(i, t)
                    .map_err(|e| EngineError::Internal(format!("token_prob {i}/{t}: {e}")))?;
                #[allow(clippy::float_arithmetic)]
                {
                    token_prob_sum += p;
                }
                token_count = token_count.saturating_add(1);
            }
        }
        let confidence = if token_count > 0 {
            // token_count is i32; as f32 loses precision for very long utterances
            // (> 16 M tokens) — acceptable for a confidence average.
            #[allow(clippy::float_arithmetic, clippy::as_conversions, clippy::cast_precision_loss)]
            let avg = token_prob_sum / token_count as f32;
            avg.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Ok(EngineFinal { text, confidence })
    }

    fn reload_model(&mut self, name: &str) -> Result<u64, EngineError> {
        validate_model_name(name).map_err(EngineError::from)?;
        let path = model_file_path(&self.models_root, name);
        let start = Instant::now();
        let new_ctx = open_context(&path)?;
        let warmup_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut guard = self
            .ctx
            .lock()
            .map_err(|e| EngineError::Internal(format!("ctx poisoned: {e}")))?;
        *guard = new_ctx;
        self.model_name = name.to_string();
        self.buffer.clear();
        Ok(warmup_ms)
    }

    fn reset(&mut self) {
        self.buffer.clear();
    }
}

fn num_threads() -> i32 {
    std::thread::available_parallelism()
        .ok()
        .and_then(|n| i32::try_from(n.get()).ok())
        .unwrap_or(1)
}

/// Public helper that lets non-feature builds reference the same path
/// convention (so the bootstrap script can install models without
/// pulling in whisper-rs). When this module is compiled out the helper
/// is hidden behind the same cfg; bootstrap should hardcode the same
/// template (`whisper-<sanitised>.bin`).
#[must_use]
pub fn resolve_model_path(root: &Path, name: &str) -> PathBuf {
    model_file_path(root, name)
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
    use crate::SttError;

    #[test]
    fn resolve_model_path_sanitises_dots() {
        let p = resolve_model_path(Path::new("/srv/models"), "distil-small.en");
        assert_eq!(p, Path::new("/srv/models/whisper-distil-small-en.bin"));
    }

    #[test]
    fn resolve_model_path_no_dots_unchanged() {
        let p = resolve_model_path(Path::new("/srv/models"), "large-v3-turbo");
        assert_eq!(p, Path::new("/srv/models/whisper-large-v3-turbo.bin"));
    }

    #[test]
    fn decode_pcm_b64_rejects_empty() {
        let err = decode_pcm_b64("").expect_err("empty rejected");
        assert!(matches!(err, EngineError::BadChunk { .. }));
    }

    #[test]
    fn decode_pcm_b64_rejects_bad_base64() {
        let err = decode_pcm_b64("!!!").expect_err("bad base64 rejected");
        match err {
            EngineError::BadChunk { reason } => assert!(reason.contains("base64")),
            other => panic!("expected BadChunk, got {other:?}"),
        }
    }

    #[test]
    fn decode_pcm_b64_rejects_odd_bytes() {
        // base64 "AAA=" decodes to 2 bytes (even) — pick a 3-byte payload.
        let three_bytes = base64::engine::general_purpose::STANDARD.encode([1_u8, 2, 3]);
        let err = decode_pcm_b64(&three_bytes).expect_err("odd bytes rejected");
        match err {
            EngineError::BadChunk { reason } => assert!(reason.contains("odd byte count")),
            other => panic!("expected BadChunk, got {other:?}"),
        }
    }

    #[test]
    fn decode_pcm_b64_round_trips_i16_samples() {
        let samples_i16: [i16; 4] = [0, i16::MAX, i16::MIN, -1];
        let mut bytes = Vec::with_capacity(samples_i16.len() * 2);
        for s in samples_i16 {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let out = decode_pcm_b64(&encoded).expect("decodes");
        assert_eq!(out.len(), samples_i16.len());
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        // i16::MIN normalises to slightly less than -1.0 (range asymmetry).
        assert!(out[2] < -1.0 + 1e-6 && out[2] > -1.01);
        assert!((out[3] + 1.0 / f32::from(i16::MAX)).abs() < 1e-6);
    }

    #[test]
    fn load_rejects_unknown_model() {
        let err =
            WhisperEngine::load("tiny.en", "/nonexistent").expect_err("unknown model rejected");
        match err {
            EngineError::UnknownModel(SttError::UnknownModel { name, .. }) => {
                assert_eq!(name, "tiny.en");
            }
            other => panic!("expected UnknownModel, got {other:?}"),
        }
    }

    /// AC6 — missing model file produces [`EngineError::ModelMissing`].
    ///
    /// `WhisperEngine::load` with a valid model name but a path that does
    /// not exist must return `ModelMissing` rather than a generic
    /// `Internal` error. This lets the daemon publish
    /// `wm.stt.error { kind: "model_missing" }` on the first speech.end
    /// instead of an opaque internal failure.
    #[test]
    fn load_missing_model_file_returns_model_missing() {
        let err = WhisperEngine::load("distil-small.en", "/nonexistent_root")
            .expect_err("missing model file rejected");
        match err {
            EngineError::ModelMissing { path } => {
                assert!(
                    path.contains("whisper-distil-small-en.bin"),
                    "path should reference the resolved model filename, got: {path}"
                );
            }
            other => panic!("expected ModelMissing, got {other:?}"),
        }
    }

    /// The `ModelMissing` variant maps to `"model_missing"` kind in error events.
    #[test]
    fn model_missing_kind_in_error_event() {
        use crate::processor;
        use crate::bus::ErrorEvent;

        let err = EngineError::ModelMissing {
            path: "/usr/share/wintermute/models/whisper-distil-small-en.bin".to_string(),
        };
        // Call the private helper via the crate-internal path (module is accessible in tests).
        let ev: ErrorEvent = processor::engine_error_event_pub(&err, 42);
        assert_eq!(ev.kind, "model_missing");
        assert_eq!(ev.ts, 42);
    }
}
