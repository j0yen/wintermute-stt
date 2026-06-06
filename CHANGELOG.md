# Changelog

## v0.4.0 — 2026-06-05

Turn-id propagation (PRD lucid-turn-id, AC3/AC5 — wm-stt leg). `wm-stt` now
copies the cross-daemon `turn_id` minted at wake by `wm-audio` from the inbound
`wm.audio.speech.end` onto every event it emits for that turn, so a consumer can
reconstruct the turn by id instead of guessing on wall-clock timestamps.

### Changes

- **Inbound `turn_id`** (`bus.rs`): `SpeechEndEvent` gains an optional
  `turn_id: Option<String>`; the field is skipped when absent so pre-PRD
  envelopes still deserialize unchanged (AC5).
- **Outbound propagation** (`bus.rs`, `processor.rs`, `daemon.rs`): `FinalEvent`,
  `UncertainEvent`, and `PartialEvent` each carry the inbound `turn_id`.
  `UtteranceProcessor` threads it through `State::Speaking` so `on_speech_end`
  copies it onto `Final`/`Uncertain` and `on_speech_chunk` onto `Partial`.
- **Tests**: `dispatch_final_carries_inbound_turn_id` (AC3: in-id == out-id),
  `dispatch_final_no_turn_id_when_absent` and
  `speech_end_legacy_no_turn_id_deserializes` (AC5: legacy/absent stays absent),
  plus roundtrip coverage for the new fields.

## v0.3.0 — 2026-06-02

Window validation, ModelMissing error kind, WM_STT_CONFIDENCE env var,
and whisper-rs build fix (PRD-wintermute-stt-whisper-model AC1/AC5/AC6/AC7/AC9).

### Changes

- **Window validation** (`processor.rs`): speech windows shorter than 200 ms or
  longer than 30 s are now rejected without inference and publish
  `wm.stt.uncertain { reason: "window_invalid", confidence: 0.0 }`. Covers AC5.
- **`EngineError::ModelMissing`** (`engine.rs`, `whisper_engine.rs`): missing model
  file now returns a typed error (rather than `Internal`) which the processor maps
  to `wm.stt.error { kind: "model_missing" }`. Covers AC6.
- **`UncertainEvent.reason`** (`bus.rs`): new optional field distinguishes
  window-validity rejects from normal low-confidence finals.
- **`WM_STT_CONFIDENCE`** (`lib.rs`): canonical env var for the confidence
  threshold; `WM_STT_THRESHOLD` still accepted as a backwards-compatible alias.
  Covers AC9.
- **whisper-rs build fix** (`.cargo/config.toml`): sets
  `WHISPER_DONT_GENERATE_BINDINGS=1` so the pre-shipped `bindings.rs` is used
  rather than regenerating them at build time (which produces a struct layout
  incompatible with whisper-rs 0.13). Enables `cargo test --lib --features whisper`.
- **Confidence via token probabilities** (`whisper_engine.rs`): `finalise` now
  averages per-token probabilities (`full_get_token_prob`) for the confidence
  score; `full_get_segment_no_speech_prob` was not available in whisper-rs 0.13.
- **Test count** (`--features whisper`): 67 tests (baseline 53, +14). Covers AC1.
- **`tests/fixtures/hello_world.wav`**: minimal 16 kHz mono WAV for future harness
  tests. Model bytes excluded via `.gitignore`. Covers AC8.
- **`install.sh --download-model`**: new flag downloads model from HuggingFace
  (`ggerganov/whisper.cpp`). Apache 2.0 licensed. Covers AC3/AC8.
- **`deny.toml`**: documented whisper-rs transitive advisory + license
  exceptions. `whisper-rs` 0.13.2 and `whisper-rs-sys` 0.11.1 publish under
  the `Unlicense` (public-domain-equivalent, OSI-approved permissive), which
  is now in the `[licenses] allow` list. `cargo deny --all-features check
  bans licenses sources` is clean. Covers AC7.

### deny.toml whisper-rs advisory + license exceptions

- License: `Unlicense` added to `[licenses] allow` for `whisper-rs`
  0.13.2 / `whisper-rs-sys` 0.11.1 (only pulled under the opt-in
  `whisper` feature).
- Advisories: no active advisories for whisper-rs 0.13 transitive deps as of
  2026-06-03. If `cargo deny check` reports `RUSTSEC-*` for whisper-rs-sys or
  ggml, add the advisory ID to `deny.toml [advisories] ignore` with a
  rationale comment.

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
