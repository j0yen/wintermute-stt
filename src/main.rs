//! `wm-stt` — wintermute speech-to-text daemon.
//!
//! iter-1 scaffold: prints version + intended responsibilities, then
//! exits with code 2 (stub). The daemon loop, agorabus subscriber, and
//! whisper.cpp inference path land in subsequent iterations per the
//! PRD §2.

use std::process::ExitCode;

#[allow(clippy::print_stderr, reason = "iter-1 scaffold stub; replaced by tracing in iter-2 daemon wiring")]
fn main() -> ExitCode {
    let version = env!("CARGO_PKG_VERSION");
    eprintln!("wm-stt {version}: scaffold stub");
    eprintln!("  intended responsibilities (per PRD-wintermute-stt §2):");
    eprintln!("    - subscribe to wm.audio.speech.* events on agorabus");
    eprintln!("    - run whisper.cpp inference on streamed PCM");
    eprintln!("    - publish wm.stt.partial / wm.stt.final / wm.stt.uncertain");
    eprintln!("  iter-1 wires nothing yet; build green is the only goal.");
    ExitCode::from(2)
}
