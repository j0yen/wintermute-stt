# wintermute-stt

Speech-to-text daemon for the wintermute voice assistant.

`wm-stt` consumes PCM speech chunks from `wm-audio` over agorabus
(`wm.audio.speech.*`) and turns them into text using **whisper.cpp**
via the `whisper-rs` Rust binding. Default model is `distil-small.en`
— the realistic CPU choice for an older laptop, with ≤2 s warm
transcription on a 5-second utterance. Confidence is emitted with
every final transcript; below threshold, the brain asks for a repeat.
An optional cloud fast-path routes to the Whisper API when network is
healthy and the user opted in during bootstrap.

This is the **STT** component of Fleet 1 of the wintermute vision.

## What it does

On startup, `wm-stt`:

- Loads the configured whisper.cpp model
  (default: `distil-small.en` at
  `/usr/share/wintermute/models/whisper-distil-small-en.bin`).
- Subscribes to agorabus topics `wm.audio.speech.*`
  (inbound from `wm-audio`) and `wm.stt.*` (operator commands like
  `reload_model`).
- On `wm.audio.speech.start`: opens a transcription session.
- On `wm.audio.speech.chunk`: feeds PCM into the engine, emits
  `wm.stt.partial` every ~500 ms.
- On `wm.audio.speech.end`: finalises the transcription off-thread
  (tokio blocking pool), computes confidence, and emits either
  `wm.stt.final` or `wm.stt.uncertain` (below the configured
  threshold, default 0.45).

`wm-stt reload-model <name>` hot-swaps the active whisper model:
in-flight transcription completes first; the new model warms up;
`wm.stt.model_loaded` is emitted on completion. Allowed names:
`distil-small.en`, `small.en`, `medium.en`, `large-v3-turbo`.

## Events published

| Topic | Payload |
|---|---|
| `wm.stt.partial` | `{text, ts}` |
| `wm.stt.final` | `{text, confidence, duration_ms, model, ts}` |
| `wm.stt.uncertain` | `{text, confidence, ts}` (below threshold) |
| `wm.stt.error` | `{kind, message, ts}` |
| `wm.stt.model_loaded` | `{model, warmup_ms, ts}` |

## Acceptance tests

1. Warm `distil-small.en` transcription of a 5-second utterance
   completes in ≤2 s on this laptop's CPU.
2. `wm.stt.partial` events emit at ~500 ms cadence during active
   speech.
3. Confidence below 0.45 emits `wm.stt.uncertain` instead of
   `wm.stt.final`; threshold is configurable.
4. With `WM_CLOUD_STT_FASTPATH=true` and network up, end-to-end
   transcription round-trip ≤500 ms for a 5-second utterance.
5. Network drop during cloud fast-path falls back to local result
   without dropping the in-flight utterance (no double-firing of
   `wm.stt.final`).
6. `wm-stt reload-model small.en` completes in <5 s without dropping
   the agorabus subscription.
7. 60-minute steady-state run shows RSS growth <50 MB (no leak).
8. Daemon recovers from `wm-audio` restart by re-subscribing within
   5 s.

Coverage: 53 lib unit tests + 1 acceptance template + 1 proptest, all
green on `cargo test --release` (default features). The `whisper`
feature requires cmake + the whisper.cpp toolchain and is exercised
on hosts where the model `.bin` files are available under
`/usr/share/wintermute/models/`.

## Install

One-liner (curl-pipe):

```
curl -fsSL https://raw.githubusercontent.com/j0yen/wintermute-stt/main/install.sh | bash
```

Or from a checkout:

```
git clone https://github.com/j0yen/wintermute-stt
cd wintermute-stt
./install.sh
```

For real inference, build with the `whisper` feature (requires cmake
and a C/C++ toolchain to compile `whisper.cpp`):

```
cargo install --path . --locked --features whisper
```

Then drop a model file at the expected path, e.g.:

```
sudo mkdir -p /usr/share/wintermute/models
sudo cp /path/to/whisper-distil-small-en.bin /usr/share/wintermute/models/
```

Start the daemon:

```
wm-stt start                                # uses defaults
wm-stt start --models-root /usr/share/wintermute/models
wm-stt reload-model medium.en               # hot-swap the model
```

## License

Dual-licensed under MIT or Apache-2.0 at your option.
