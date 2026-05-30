# Changelog

## v0.2.0 — 2026-05-30

Wire `WhisperEngine` into the daemon (PRD-wintermute-stt-whisper-model).

`daemon::run()` now selects the engine at compile time via a cfg gate:
when built with `--features whisper` it constructs a real `WhisperEngine`
(whisper.cpp via `whisper-rs`) backed by the model file at
`<models_root>/whisper-<name>.bin`; the default build retains `StubEngine`
so development without the whisper.cpp toolchain still compiles. The new
`build_whisper_processor` helper (feature-gated) mirrors `build_stub_processor`.
Also bumps the `agorabus` path-dep pin from `0.3` → `0.8` to match the
installed workspace version.

## v0.1.1 — 2026-05-28

Fix post-announce bus-startup defect (PRD-wintermute-fleet-bus-startup-defect).

The announce-before-subscribe fix that shipped overnight was install-stale, not
source-buggy: the binaries under ~/.local/bin/ predated the fix, while the source
already had the dual-Client + announce-first pattern. Tightened the agorabus
path-dependency pin from a wildcard/^0.1 to ^0.3 (agorabus 0.3.0's let_chains
need system cargo 1.95), rebuilt, and reinstalled so the systemd-launched daemons
run post-fix bytes. Daemons now survive a 60s soak (NRestarts=0) and round-trip
their subscribed topics. Note: AC3-strict (peer presence after the 60s window)
is deferred to PRD-wintermute-fleet-bus-heartbeat-keepalive — these daemons still
lack a post-announce heartbeat, so the bus prunes them from the peer snapshot.
