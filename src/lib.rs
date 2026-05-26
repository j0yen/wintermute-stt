//! `wintermute-stt` — speech-to-text library that backs the `wm-stt` daemon.
//!
//! iter-3 surface: runtime [`SttConfig`] (env loader + validation), the
//! [`SttError`] enum, and the agorabus topic schema in [`bus`] — typed
//! payloads for inbound `wm.audio.speech.{start,chunk,end}` +
//! `wm.stt.reload_model`, outbound `wm.stt.{partial,final,uncertain,
//! error,model_loaded}`, plus the [`bus::decode_request`] entry point.
//! The live subscribe loop and whisper.cpp engine land in iter-4+ per
//! `PRD-wintermute-stt.md` §2.

#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod bus;
pub mod daemon;
pub mod engine;
pub mod processor;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Default whisper.cpp model name when `WM_STT_MODEL` is unset.
///
/// Plan-agent's correction: `distil-small.en` is the realistic CPU
/// default on this laptop; see PRD §1.2.
pub const DEFAULT_MODEL_NAME: &str = "distil-small.en";

/// Default confidence threshold; below this, the daemon emits
/// `wm.stt.uncertain` instead of `wm.stt.final`. PRD §2.4.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.45;

/// Default on-disk root for whisper.cpp model files. The active model's
/// file is resolved as `<root>/whisper-<name>.bin`.
pub const DEFAULT_MODELS_ROOT: &str = "/usr/share/wintermute/models";

/// Default Unix socket the daemon reads PCM frames from. Owned by
/// `wm-audio`; see `PRD-wintermute-audio.md`.
pub const DEFAULT_MIC_SOCK: &str = "/run/wintermute/mic.sock";

/// Model names the daemon allows for `reload-model` and startup load.
/// PRD §2.3.
pub const ALLOWED_MODEL_NAMES: &[&str] =
    &["distil-small.en", "small.en", "medium.en", "large-v3-turbo"];

/// Runtime configuration for `wm-stt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SttConfig {
    /// Active whisper.cpp model name. Must be in [`ALLOWED_MODEL_NAMES`].
    #[serde(default = "default_model")]
    pub model: String,
    /// On-disk root for whisper model files.
    #[serde(default = "default_models_root")]
    pub models_root: PathBuf,
    /// Path to the PCM mic socket exposed by `wm-audio`.
    #[serde(default = "default_mic_sock")]
    pub mic_sock: PathBuf,
    /// Confidence threshold. Must be within `(0.0, 1.0]`.
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f32,
    /// When true, also stream chunks to the Whisper cloud API; whichever
    /// returns first emits the final event. PRD §2.2.
    #[serde(default)]
    pub cloud_fastpath: bool,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            models_root: default_models_root(),
            mic_sock: default_mic_sock(),
            confidence_threshold: default_threshold(),
            cloud_fastpath: false,
        }
    }
}

fn default_model() -> String {
    DEFAULT_MODEL_NAME.to_string()
}

fn default_models_root() -> PathBuf {
    PathBuf::from(DEFAULT_MODELS_ROOT)
}

fn default_mic_sock() -> PathBuf {
    PathBuf::from(DEFAULT_MIC_SOCK)
}

const fn default_threshold() -> f32 {
    DEFAULT_CONFIDENCE_THRESHOLD
}

/// Errors raised by config loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum SttError {
    /// I/O error reading a file from disk.
    #[error("io failure on {path}: {source}")]
    Io {
        /// Path that produced the error.
        path: PathBuf,
        /// Underlying I/O cause.
        #[source]
        source: std::io::Error,
    },
    /// Model name not in [`ALLOWED_MODEL_NAMES`].
    #[error("unknown model {name:?}; allowed: {allowed:?}")]
    UnknownModel {
        /// Offending model name.
        name: String,
        /// Allow-list at the moment of the failure.
        allowed: Vec<String>,
    },
    /// Confidence threshold outside `(0.0, 1.0]`.
    #[error("invalid confidence threshold {0}; must be in (0.0, 1.0]")]
    InvalidThreshold(f32),
    /// Env var present but unparseable into the expected type.
    #[error("env var {var} has invalid value {value:?}: {reason}")]
    InvalidEnv {
        /// Env var name.
        var: &'static str,
        /// Raw string value seen.
        value: String,
        /// Human-readable parse failure.
        reason: String,
    },
}

/// Reject model names that are not in [`ALLOWED_MODEL_NAMES`].
///
/// # Errors
/// Returns [`SttError::UnknownModel`] when `name` is not a member of the
/// allow-list. The error includes a snapshot of the allow-list so the
/// caller can present a useful message.
pub fn validate_model_name(name: &str) -> Result<(), SttError> {
    if ALLOWED_MODEL_NAMES.contains(&name) {
        Ok(())
    } else {
        Err(SttError::UnknownModel {
            name: name.to_string(),
            allowed: ALLOWED_MODEL_NAMES.iter().map(|s| (*s).to_string()).collect(),
        })
    }
}

/// Reject confidence thresholds outside `(0.0, 1.0]`.
///
/// # Errors
/// Returns [`SttError::InvalidThreshold`] for `NaN`, `<= 0.0`, or
/// `> 1.0`. Exactly `1.0` is accepted (means "never uncertain").
pub fn validate_threshold(t: f32) -> Result<(), SttError> {
    if t.is_nan() || t <= 0.0 || t > 1.0 {
        Err(SttError::InvalidThreshold(t))
    } else {
        Ok(())
    }
}

impl SttConfig {
    /// Build a config from environment variables, falling back to
    /// defaults for any unset variable.
    ///
    /// Recognised vars:
    /// - `WM_STT_MODEL` (string; default `distil-small.en`)
    /// - `WM_STT_MODELS_ROOT` (path; default `/usr/share/wintermute/models`)
    /// - `WM_STT_MIC_SOCK` (path; default `/run/wintermute/mic.sock`)
    /// - `WM_STT_THRESHOLD` (float; default `0.45`)
    /// - `WM_CLOUD_STT_FASTPATH` (`true`/`false`; default `false`)
    ///
    /// # Errors
    /// Returns [`SttError::InvalidEnv`] when a var is set but
    /// unparseable, or the more specific validation error when the
    /// parsed value fails [`validate_model_name`] / [`validate_threshold`].
    pub fn from_env() -> Result<Self, SttError> {
        let model = env_string("WM_STT_MODEL").unwrap_or_else(default_model);
        validate_model_name(&model)?;

        let models_root = env_string("WM_STT_MODELS_ROOT")
            .map_or_else(default_models_root, PathBuf::from);
        let mic_sock =
            env_string("WM_STT_MIC_SOCK").map_or_else(default_mic_sock, PathBuf::from);

        let confidence_threshold = match env_string("WM_STT_THRESHOLD") {
            Some(raw) => parse_threshold_env(&raw)?,
            None => default_threshold(),
        };
        validate_threshold(confidence_threshold)?;

        let cloud_fastpath = match env_string("WM_CLOUD_STT_FASTPATH") {
            Some(raw) => parse_bool_env("WM_CLOUD_STT_FASTPATH", &raw)?,
            None => false,
        };

        Ok(Self {
            model,
            models_root,
            mic_sock,
            confidence_threshold,
            cloud_fastpath,
        })
    }

    /// Validate an already-constructed config. Used after deserialising
    /// from any future config file or after CLI-flag overrides.
    ///
    /// # Errors
    /// Forwards from [`validate_model_name`] and [`validate_threshold`].
    pub fn validate(&self) -> Result<(), SttError> {
        validate_model_name(&self.model)?;
        validate_threshold(self.confidence_threshold)?;
        Ok(())
    }
}

fn env_string(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.is_empty())
}

fn parse_threshold_env(raw: &str) -> Result<f32, SttError> {
    raw.parse::<f32>().map_err(|e| SttError::InvalidEnv {
        var: "WM_STT_THRESHOLD",
        value: raw.to_string(),
        reason: e.to_string(),
    })
}

fn parse_bool_env(var: &'static str, raw: &str) -> Result<bool, SttError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(SttError::InvalidEnv {
            var,
            value: raw.to_string(),
            reason: "expected one of: 1/0, true/false, yes/no, on/off".to_string(),
        }),
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
    fn default_config_validates() {
        let cfg = SttConfig::default();
        cfg.validate().expect("defaults are valid");
        assert_eq!(cfg.model, DEFAULT_MODEL_NAME);
        assert_eq!(cfg.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD);
        assert!(!cfg.cloud_fastpath);
    }

    #[test]
    fn validate_model_name_accepts_all_allowed() {
        for name in ALLOWED_MODEL_NAMES {
            validate_model_name(name).expect("allow-list entries validate");
        }
    }

    #[test]
    fn validate_model_name_rejects_unknown() {
        let err = validate_model_name("tiny.en").unwrap_err();
        assert!(matches!(err, SttError::UnknownModel { ref name, .. } if name == "tiny.en"));
    }

    #[test]
    fn validate_threshold_boundaries() {
        validate_threshold(0.001).expect("just above zero is fine");
        validate_threshold(1.0).expect("exactly one is fine");
        assert!(matches!(
            validate_threshold(0.0),
            Err(SttError::InvalidThreshold(_))
        ));
        assert!(matches!(
            validate_threshold(-0.1),
            Err(SttError::InvalidThreshold(_))
        ));
        assert!(matches!(
            validate_threshold(1.1),
            Err(SttError::InvalidThreshold(_))
        ));
        assert!(matches!(
            validate_threshold(f32::NAN),
            Err(SttError::InvalidThreshold(_))
        ));
    }

    #[test]
    fn parse_bool_env_accepts_synonyms() {
        for raw in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(parse_bool_env("WM_TEST", raw).expect("parses"));
        }
        for raw in ["0", "false", "no", "off"] {
            assert!(!parse_bool_env("WM_TEST", raw).expect("parses"));
        }
        assert!(matches!(
            parse_bool_env("WM_TEST", "maybe"),
            Err(SttError::InvalidEnv { .. })
        ));
    }

    #[test]
    fn round_trip_serde_json() {
        let cfg = SttConfig::default();
        let v = serde_json::to_value(&cfg).expect("serialises");
        let back: SttConfig = serde_json::from_value(v).expect("round-trips");
        assert_eq!(cfg, back);
    }
}
