//! `wm-stt` CLI entrypoint.
//!
//! iter-2 wires a clap dispatcher and `tracing` initialisation. The
//! `start` subcommand resolves a [`SttConfig`] from env vars and prints
//! it (debug-level) then exits 2 — the daemon loop, whisper.cpp
//! inference, and the live agorabus subscriber land in iter-3+. The
//! `reload-model` and `transcribe` subcommands are also stubs at exit
//! code 2; they will become agorabus producers once iter-4 lands the
//! bus schema.

#![cfg_attr(not(test), forbid(unsafe_code))]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use wintermute_stt::{
    DEFAULT_MIC_SOCK, DEFAULT_MODELS_ROOT, SttConfig, validate_model_name,
};

#[derive(Parser, Debug)]
#[command(
    name = "wm-stt",
    version,
    about = "wintermute speech-to-text daemon and CLI"
)]
struct Cli {
    /// Override the whisper.cpp models root (defaults to
    /// `/usr/share/wintermute/models`).
    #[arg(long, global = true, default_value = DEFAULT_MODELS_ROOT)]
    models_root: PathBuf,

    /// Override the PCM mic socket (defaults to
    /// `/run/wintermute/mic.sock`).
    #[arg(long, global = true, default_value = DEFAULT_MIC_SOCK)]
    mic_sock: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start the daemon (long-running). iter-3 wires whisper.cpp.
    Start,
    /// Ask a running daemon to hot-swap to a different whisper model.
    ReloadModel {
        /// One of: `distil-small.en`, `small.en`, `medium.en`,
        /// `large-v3-turbo`.
        name: String,
    },
    /// One-shot transcription of a PCM file (iter-4+).
    Transcribe {
        /// Path to a 16 kHz mono PCM file.
        input: PathBuf,
    },
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    drop(tracing_subscriber::fmt().with_env_filter(filter).try_init());
}

fn main() -> ExitCode {
    init_tracing();
    let cli = Cli::parse();

    match cli.command {
        Command::Start => run_start(&cli.models_root, &cli.mic_sock),
        Command::ReloadModel { name } => run_reload_model(&name),
        Command::Transcribe { input } => run_transcribe(&input),
    }
}

fn run_start(models_root: &Path, mic_sock: &Path) -> ExitCode {
    let mut cfg = match SttConfig::from_env() {
        Ok(c) => c,
        Err(err) => {
            error!(error = %err, "wm-stt start: invalid configuration");
            return ExitCode::from(1);
        }
    };
    cfg.models_root = models_root.to_path_buf();
    cfg.mic_sock = mic_sock.to_path_buf();
    if let Err(err) = cfg.validate() {
        error!(error = %err, "wm-stt start: config validation failed");
        return ExitCode::from(1);
    }
    info!(?cfg, "wm-stt start: config resolved");
    warn!("wm-stt start: daemon loop deferred to iter-3");
    ExitCode::from(2)
}

fn run_reload_model(name: &str) -> ExitCode {
    if let Err(err) = validate_model_name(name) {
        error!(error = %err, "wm-stt reload-model: unknown model");
        return ExitCode::from(1);
    }
    warn!(model = %name, "wm-stt reload-model: agorabus publish deferred to iter-4");
    ExitCode::from(2)
}

fn run_transcribe(input: &Path) -> ExitCode {
    warn!(input = %input.display(), "wm-stt transcribe: deferred to iter-4");
    ExitCode::from(2)
}
