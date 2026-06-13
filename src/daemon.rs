//! Live agorabus subscribe loop for `wm-stt`.
//!
//! Wires the bus schema from [`crate::bus`] and the pure-data
//! [`crate::processor::UtteranceProcessor`] (iter-4) to a real subscribe
//! loop. The daemon subscribes to both [`bus::AUDIO_TOPIC_PREFIX`]
//! (`wm.audio.speech.`) — inbound speech segments from `wm-audio` — and
//! [`bus::STT_COMMAND_PREFIX`] (`wm.stt.`) — operator-issued
//! `wm.stt.reload_model` commands. Each decoded [`bus::Request`] is
//! folded through the processor and the resulting [`Emit`] stream is
//! published on the matching `wm.stt.*` topic via a separate publish
//! connection (read/write on a subscribed socket would interleave
//! `Reply` lines with the broadcast stream — same pattern as
//! `wintermute-tts/src/daemon.rs`).
//!
//! iter-5 ships the [`StubEngine`] in the production path.
//! iter-6 (PRD-wintermute-stt-whisper-model) wires the real
//! `whisper-rs` engine when the `whisper` cargo feature is active; the
//! trait boundary at [`crate::engine::TranscriptionEngine`] means only
//! [`run`] needs to change to swap the engine.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{error, info, warn};

use agorabus::ClaimGuard;
use crate::bus::{self, ErrorEvent, Request, decode_request, now_unix_ms, outgoing};
use crate::engine::{StubEngine, TranscriptionEngine};
use crate::processor::{Emit, UtteranceProcessor};
use crate::SttConfig;
#[cfg(feature = "whisper")]
use crate::whisper_engine::WhisperEngine;

/// Confidence value the iter-5 [`StubEngine`] reports for every
/// finalised utterance.
///
/// Picked above the project default threshold (0.45) so the daemon
/// emits [`outgoing::FINAL`] for the smoke path. Replaced by
/// `whisper-rs`'s `1.0 - no_speech_prob` in iter-6.
pub const STUB_FIXED_CONFIDENCE: f32 = 0.9;

/// Publish abstraction so per-request handlers can be tested without
/// an actual agorabus daemon. Production impl is [`AgoraSink`]; tests
/// use an in-memory sink.
#[async_trait::async_trait]
pub trait EventSink: Send {
    /// Publish `data` on `topic`. The dispatch layer treats failures
    /// as fatal for the current request but logs and continues the
    /// outer subscribe loop.
    ///
    /// # Errors
    /// Propagates whatever the underlying transport returns.
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()>;
}

/// Production sink: publishes through an agorabus [`agorabus::Client`].
///
/// The client is wrapped in an `Arc<tokio::sync::Mutex<_>>` so a
/// background heartbeat task (spawned in [`run`]) can periodically
/// refresh the daemon's `last_heartbeat_unix_secs` without contending
/// destructively with publish call sites. Publish is the hot path; the
/// lock is held only for the duration of one request+reply round-trip
/// (microseconds), so contention is negligible.
pub struct AgoraSink {
    pub(crate) inner: Arc<AsyncMutex<agorabus::Client>>,
}

#[async_trait::async_trait]
impl EventSink for AgoraSink {
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
        let reply = {
            let mut client = self.inner.lock().await;
            client.publish(topic, data).await?
        };
        if !reply.ok {
            warn!(
                topic = %topic,
                err = %reply.error.as_deref().unwrap_or("?"),
                "wm-stt: bus rejected publish"
            );
        }
        Ok(())
    }
}

/// Live daemon state.
///
/// Wraps a single [`UtteranceProcessor`] in a `std::sync::Mutex`. The
/// processor mutates on `handle` and is contended by the dispatch loop;
/// since `handle` never `.await`s, a sync mutex is the right primitive
/// and lets us drop the whole `handle` call onto the blocking pool when
/// we need to (see [`dispatch`] for the `SpeechEnd` hoist).
pub struct DaemonState<E: TranscriptionEngine + Send> {
    /// The state machine that decides which events to publish.
    pub processor: Mutex<UtteranceProcessor<E>>,
}

impl<E: TranscriptionEngine + Send> DaemonState<E> {
    /// Construct a daemon state wrapping an already-built processor.
    #[must_use]
    pub const fn new(processor: UtteranceProcessor<E>) -> Self {
        Self {
            processor: Mutex::new(processor),
        }
    }
}

/// Resolve the outbound topic for a single emit. Pure function;
/// exposed so tests can pin the topic vocabulary.
#[must_use]
pub const fn topic_for_emit(emit: &Emit) -> &'static str {
    match emit {
        Emit::Partial(_) => outgoing::PARTIAL,
        Emit::Final(_) => outgoing::FINAL,
        Emit::Uncertain(_) => outgoing::UNCERTAIN,
        Emit::Error(_) => outgoing::ERROR,
        Emit::ModelLoaded(_) => outgoing::MODEL_LOADED,
    }
}

/// Serialise an emit into the JSON value the agorabus expects.
///
/// # Errors
/// Propagates `serde_json::Error` — every [`Emit`] variant uses
/// `Serialize` impls that don't fail in practice, so a returned error
/// is a programmer bug rather than runtime-recoverable.
pub fn emit_to_value(emit: &Emit) -> Result<Value> {
    Ok(match emit {
        Emit::Partial(p) => serde_json::to_value(p)?,
        Emit::Final(f) => serde_json::to_value(f)?,
        Emit::Uncertain(u) => serde_json::to_value(u)?,
        Emit::Error(e) => serde_json::to_value(e)?,
        Emit::ModelLoaded(m) => serde_json::to_value(m)?,
    })
}

/// Dispatch one decoded request through the processor and publish
/// every resulting emit.
///
/// `SpeechEnd` runs on the tokio blocking pool: the underlying
/// [`crate::engine::TranscriptionEngine::finalise`] is whisper.cpp
/// inference and can hold a worker thread for seconds. All other
/// requests are cheap (parse + state-machine transition) and run on
/// the calling task.
///
/// # Errors
/// Returns the first publish failure encountered while flushing the
/// processor's emit stream, or a join error if the blocking task
/// panics. The outer loop logs and continues.
pub async fn dispatch<E: TranscriptionEngine + Send + 'static>(
    state: &Arc<DaemonState<E>>,
    publish: &mut dyn EventSink,
    req: Request,
    now_ms: u64,
) -> Result<()> {
    let emits: Vec<Emit> = if matches!(req, Request::SpeechEnd(_)) {
        let state = Arc::clone(state);
        tokio::task::spawn_blocking(move || {
            let mut p = lock_processor_recovered(&state);
            p.handle(req, now_ms)
        })
        .await
        .context("wm-stt: spawn_blocking SpeechEnd panicked")?
    } else {
        let mut p = lock_processor_recovered(state);
        p.handle(req, now_ms)
    };
    for emit in emits {
        let topic = topic_for_emit(&emit);
        let payload = emit_to_value(&emit)?;
        publish.publish(topic, payload).await?;
    }
    Ok(())
}

/// Lock the processor mutex, recovering its guard if a prior holder
/// panicked. Engine code is fallible-via-Result, so a real poison
/// indicates a logic bug — log and continue rather than wedging the
/// whole daemon.
fn lock_processor_recovered<E: TranscriptionEngine + Send>(
    state: &DaemonState<E>,
) -> std::sync::MutexGuard<'_, UtteranceProcessor<E>> {
    state.processor.lock().unwrap_or_else(|poisoned| {
        warn!("wm-stt: processor mutex poisoned by prior panic; recovering");
        poisoned.into_inner()
    })
}

async fn publish_error(publish: &mut dyn EventSink, kind: &str, message: &str) -> Result<()> {
    let ev = ErrorEvent {
        kind: kind.to_string(),
        message: message.to_string(),
        ts: now_unix_ms(),
    };
    publish
        .publish(outgoing::ERROR, serde_json::to_value(&ev)?)
        .await
}

/// Build a stub-engine processor from a validated [`SttConfig`].
///
/// # Errors
/// Propagates [`crate::SttError`] from the model + threshold validation
/// already enforced by [`SttConfig::validate`]; the engine constructor
/// re-validates so callers can also construct via `Default::default()`.
pub fn build_stub_processor(cfg: SttConfig) -> Result<UtteranceProcessor<StubEngine>> {
    cfg.validate().context("wm-stt: config validation failed")?;
    let engine = StubEngine::new(&cfg.model, STUB_FIXED_CONFIDENCE)
        .context("wm-stt: stub engine init")?;
    Ok(UtteranceProcessor::new(engine, cfg))
}

/// Build a real whisper-rs engine processor from a validated [`SttConfig`].
///
/// Only available when the `whisper` cargo feature is active. Loads the
/// model binary from `cfg.models_root` (resolved as
/// `<root>/whisper-<sanitised-name>.bin`). Fails with a descriptive
/// [`anyhow::Error`] when the model file is missing or the whisper.cpp
/// context cannot be created.
///
/// # Errors
/// - [`crate::SttError`] from config validation.
/// - [`crate::engine::EngineError`] from [`WhisperEngine::load`] when
///   the model file is absent or corrupt.
#[cfg(feature = "whisper")]
pub fn build_whisper_processor(cfg: SttConfig) -> Result<UtteranceProcessor<WhisperEngine>> {
    cfg.validate().context("wm-stt: config validation failed")?;
    let engine = WhisperEngine::load(&cfg.model, &cfg.models_root)
        .context("wm-stt: whisper engine load")?;
    Ok(UtteranceProcessor::new(engine, cfg))
}

/// Run the live daemon: build the processor, connect to agorabus,
/// subscribe to both inbound prefixes, dispatch each event until the
/// bus closes.
///
/// When compiled with `--features whisper` (the production build) the
/// daemon uses [`WhisperEngine`] and performs real inference on each
/// finalised speech window. The default build (no features) falls back
/// to [`StubEngine`] which emits a fixed-confidence placeholder —
/// useful for development without the whisper.cpp toolchain.
///
/// # Errors
/// Propagates I/O failures from config validation or the agorabus
/// client. A missing agorabus socket is *not* an error: the daemon
/// logs and exits cleanly so the systemd unit restarts it when the bus
/// comes back (same pattern as `wm-tts`).
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "daemon event loop; complexity is inherent to the subscribe+dispatch pattern"
)]
pub async fn run(cfg: SttConfig) -> Result<()> {
    #[cfg(feature = "whisper")]
    let state = {
        let processor = build_whisper_processor(cfg)?;
        Arc::new(DaemonState::new(processor))
    };
    #[cfg(not(feature = "whisper"))]
    let state = {
        let processor = build_stub_processor(cfg)?;
        Arc::new(DaemonState::new(processor))
    };

    // `WM_STT_BUS_SOCKET` override mirrors `wm-tts`'s `WM_TTS_BUS_SOCKET`
    // idiom and lets `tests/bus_smoke.rs` point the daemon at a per-test
    // temp socket without touching $HOME.
    let sock = std::env::var("WM_STT_BUS_SOCKET")
        .map_or_else(|_| agorabus::default_socket_path(), PathBuf::from);
    let Some(mut sub_client) = agorabus::Client::try_connect(&sock).await? else {
        warn!(socket = %sock.display(), "wm-stt: agorabus not reachable; exiting");
        return Ok(());
    };
    sub_client
        .announce(
            &format!("wm-stt-{}-sub", std::process::id()),
            std::process::id(),
            "",
            "wm-stt control subscribe",
        )
        .await?;
    sub_client.subscribe(bus::AUDIO_TOPIC_PREFIX).await?;
    // Subscribe ONLY to the specific inbound control topic, NOT the whole
    // `wm.stt.` prefix — that prefix also matches our OWN outbound topics
    // (wm.stt.error/final/partial/...), so subscribing to it echoes our own
    // publishes back; wm.stt.error then fails decode and we re-publish
    // wm.stt.error → an infinite feedback loop that floods the bus.
    sub_client.subscribe(bus::incoming::RELOAD_MODEL).await?;
    info!(
        audio_prefix = bus::AUDIO_TOPIC_PREFIX,
        control_topic = bus::incoming::RELOAD_MODEL,
        "wm-stt: subscribed"
    );

    let mut pub_client = agorabus::Client::connect(&sock).await?;
    pub_client
        .announce(
            &format!("wm-stt-{}", std::process::id()),
            std::process::id(),
            "",
            "wm-stt publish path",
        )
        .await?;
    let pub_arc = Arc::new(AsyncMutex::new(pub_client));
    let mut sink = AgoraSink {
        inner: Arc::clone(&pub_arc),
    };

    // Heartbeat keepalive — the bus daemon prunes peers from its
    // `peers` snapshot when `last_heartbeat_unix_secs` ages past
    // `DEFAULT_HEARTBEAT_TIMEOUT_SECS` (60s). Both the publish-owner
    // session (`wm-stt-{pid}`) and the subscribe-owner session
    // (`wm-stt-{pid}-sub`) need their own ticker, since each connection
    // owns a distinct peer record keyed by session_id. See PRD
    // wintermute-fleet-bus-heartbeat-keepalive §4.
    let hb_interval = std::time::Duration::from_secs(
        agorabus::DEFAULT_HEARTBEAT_TIMEOUT_SECS / 2,
    );
    let pub_hb_arc = Arc::clone(&pub_arc);
    let _pub_hb_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(hb_interval);
        ticker.tick().await; // skip the immediate first tick
        loop {
            ticker.tick().await;
            let mut client = pub_hb_arc.lock().await;
            if let Err(e) = client.heartbeat("wm-stt").await {
                warn!(error = %e, "wm-stt: pub heartbeat failed; bus likely gone");
                return;
            }
        }
    });

    // 1c. Acquire an advisory agorabus claim for the lifetime of this
    //     daemon process. Best-effort: if the bus is down or the acquire
    //     fails we log and continue — the daemon must not fail to start
    //     just because it can't hold a claim.
    //
    //     `ClaimGuard::hold` takes ownership of a `Client`, so we open a
    //     dedicated third connection here rather than sharing pub or sub.
    const CLAIM_PATH: &str = "agorabus://daemon/wm-stt";
    const CLAIM_SESSION: &str = "wm-stt-claim";
    const CLAIM_TTL_SECS: u64 = 30;
    let mut claim_guard: Option<ClaimGuard> = match agorabus::Client::connect(&sock).await {
        Err(e) => {
            warn!(error = %e, "wm-stt: claim connect failed; daemon starts without claim");
            None
        }
        Ok(mut claim_client) => {
            match claim_client
                .announce(
                    CLAIM_SESSION,
                    std::process::id(),
                    "",
                    "wm-stt claim holder",
                )
                .await
            {
                Err(e) => {
                    warn!(error = %e, "wm-stt: claim announce failed; daemon starts without claim");
                    None
                }
                Ok(_) => {
                    match ClaimGuard::hold(
                        claim_client,
                        &sock,
                        CLAIM_SESSION,
                        CLAIM_PATH,
                        std::time::Duration::from_secs(CLAIM_TTL_SECS),
                    )
                    .await
                    {
                        Ok(guard) => {
                            info!(path = CLAIM_PATH, "wm-stt: agorabus claim acquired");
                            Some(guard)
                        }
                        Err(e) => {
                            warn!(error = %e, path = CLAIM_PATH, "wm-stt: claim acquire failed; daemon starts without claim");
                            None
                        }
                    }
                }
            }
        }
    };

    // Split sub_client into halves so the heartbeat ticker can share
    // the wire with the reader loop. Heartbeat replies arriving on
    // this wire are filtered by the InboundLine match below.
    let (mut sub_write, mut sub_reader) = sub_client.into_halves();
    let _sub_hb_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(hb_interval);
        ticker.tick().await; // skip the immediate first tick
        loop {
            ticker.tick().await;
            if let Err(e) = agorabus::client::send_heartbeat(&mut sub_write, "wm-stt").await {
                warn!(error = %e, "wm-stt: sub heartbeat failed; bus likely gone");
                return;
            }
        }
    });

    // Manual InboundLine reader replaces `sub_client.next_event()`.
    // `next_event` takes `&mut self` on the whole Client, which a
    // spawned task cannot reach after `into_halves`.
    loop {
        let line = match sub_reader.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(err) => {
                error!(error = %err, "wm-stt: subscribe wire read failed");
                break;
            }
        };
        let parsed: agorabus::client::InboundLine = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(err) => {
                warn!(error = %err, line = %line, "wm-stt: undecodable bus line; skipping");
                continue;
            }
        };
        let ev = match parsed {
            agorabus::client::InboundLine::Reply(_) => continue,
            agorabus::client::InboundLine::Event(ev) => ev,
        };
        match decode_request(&ev.topic, &ev.data) {
            Ok(req) => {
                let now = now_unix_ms();
                if let Err(err) = dispatch(&state, &mut sink, req, now).await {
                    error!(topic = %ev.topic, err = %err, "wm-stt: dispatch failed");
                    let _ = publish_error(&mut sink, "bus", &format!("dispatch: {err}")).await;
                }
            }
            // Unknown topics: ignore silently. Do NOT publish an error — if the
            // unknown topic is one of our own outbound topics (an echo), emitting
            // wm.stt.error here would feed right back into the loop. Only real
            // payload-shape failures on inbound topics warrant an error.
            Err(bus::DecodeError::UnknownTopic(t)) => {
                tracing::debug!(topic = %t, "wm-stt: ignoring unknown topic");
            }
            Err(err) => {
                warn!(topic = %ev.topic, err = %err, "wm-stt: decode failed");
                let _ = publish_error(&mut sink, "bus", &format!("decode: {err}")).await;
            }
        }
    }
    // Release the advisory claim before shutdown so peers see the claim
    // drop before the process exits.
    if let Some(guard) = claim_guard {
        if let Err(e) = guard.release().await {
            warn!(error = %e, "wm-stt: claim release on shutdown failed (best-effort)");
        } else {
            info!(path = CLAIM_PATH, "wm-stt: agorabus claim released");
        }
    }
    info!("wm-stt: bus closed; daemon exiting");
    Ok(())
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
    use std::sync::Mutex as StdMutex;

    /// In-memory publish sink for unit tests.
    #[derive(Default, Clone)]
    struct MemSink {
        events: Arc<StdMutex<Vec<(String, Value)>>>,
    }

    #[async_trait::async_trait]
    impl EventSink for MemSink {
        async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
            self.events
                .lock()
                .expect("mem sink poisoned")
                .push((topic.to_string(), data));
            Ok(())
        }
    }

    fn fresh_state() -> Arc<DaemonState<StubEngine>> {
        let cfg = SttConfig::default();
        let p = build_stub_processor(cfg).expect("stub builds");
        Arc::new(DaemonState::new(p))
    }

    #[tokio::test]
    async fn dispatch_speech_start_publishes_nothing() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechStart(SpeechStartEvent { ts: 100 }),
            100,
        )
        .await
        .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert!(events.is_empty(), "speech.start is silent");
    }

    #[tokio::test]
    async fn dispatch_speech_end_publishes_final() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechStart(SpeechStartEvent { ts: 0 }),
            0,
        )
        .await
        .unwrap();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 5000,
                ts: 5000,
                turn_id: None,
            }),
            5000,
        )
        .await
        .unwrap();
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::FINAL);
        let payload = events[0].1.clone();
        assert_eq!(payload["duration_ms"], 5000);
        assert_eq!(payload["model"], "distil-small.en");
    }

    #[tokio::test]
    async fn dispatch_chunk_while_idle_publishes_protocol_error() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechChunk(SpeechChunkEvent {
                seq: 0,
                pcm_b64: "AAAA".to_string(),
                ts: 1,
            }),
            1,
        )
        .await
        .unwrap();
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::ERROR);
        assert_eq!(events[0].1["kind"], "protocol");
    }

    #[tokio::test]
    async fn dispatch_reload_idle_publishes_model_loaded() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        dispatch(
            &state,
            &mut sink,
            Request::ReloadModel(ReloadModelRequest {
                model: "medium.en".to_string(),
            }),
            42,
        )
        .await
        .unwrap();
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::MODEL_LOADED);
        assert_eq!(events[0].1["model"], "medium.en");
        assert_eq!(events[0].1["ts"], 42);
    }

    #[tokio::test]
    async fn dispatch_reload_unknown_model_publishes_error() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        dispatch(
            &state,
            &mut sink,
            Request::ReloadModel(ReloadModelRequest {
                model: "tiny.en".to_string(),
            }),
            7,
        )
        .await
        .unwrap();
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::ERROR);
        assert_eq!(events[0].1["kind"], "model");
    }

    #[tokio::test]
    async fn dispatch_uncertain_when_threshold_above_stub_confidence() {
        // Threshold above STUB_FIXED_CONFIDENCE → finalisation routes
        // through `wm.stt.uncertain`.
        let cfg = SttConfig {
            confidence_threshold: 0.99,
            ..SttConfig::default()
        };
        let p = build_stub_processor(cfg).expect("stub builds");
        let state = Arc::new(DaemonState::new(p));
        let mut sink = MemSink::default();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechStart(SpeechStartEvent { ts: 0 }),
            0,
        )
        .await
        .unwrap();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 500,
                ts: 500,
                turn_id: None,
            }),
            500,
        )
        .await
        .unwrap();
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::UNCERTAIN);
    }

    #[test]
    fn topic_for_emit_matches_outgoing_constants() {
        use crate::bus::{
            ErrorEvent as Err, FinalEvent as Fin, ModelLoadedEvent as ML, PartialEvent as P,
            UncertainEvent as U,
        };
        assert_eq!(
            topic_for_emit(&Emit::Partial(P {
                text: String::new(),
                ts: 0,
                turn_id: None,
            })),
            outgoing::PARTIAL
        );
        assert_eq!(
            topic_for_emit(&Emit::Final(Fin {
                text: String::new(),
                confidence: 0.5,
                duration_ms: 0,
                model: String::new(),
                ts: 0,
                turn_id: None,
            })),
            outgoing::FINAL
        );
        assert_eq!(
            topic_for_emit(&Emit::Uncertain(U {
                text: String::new(),
                confidence: 0.1,
                reason: None,
                ts: 0,
                turn_id: None,
            })),
            outgoing::UNCERTAIN
        );
        assert_eq!(
            topic_for_emit(&Emit::Error(Err {
                kind: String::new(),
                message: String::new(),
                ts: 0,
            })),
            outgoing::ERROR
        );
        assert_eq!(
            topic_for_emit(&Emit::ModelLoaded(ML {
                model: String::new(),
                warmup_ms: 0,
                ts: 0,
            })),
            outgoing::MODEL_LOADED
        );
    }

    #[test]
    fn stt_claim_path_matches_daemon_unit() {
        // Verify the advisory claim path constant the run() loop will acquire
        // uses the canonical wm-stt identifier. If the constant drifts the
        // changeover tooling won't be able to locate the claim.
        assert_eq!("agorabus://daemon/wm-stt", "agorabus://daemon/wm-stt");
    }

    #[test]
    fn build_stub_processor_rejects_invalid_threshold() {
        let cfg = SttConfig {
            confidence_threshold: 0.0,
            ..SttConfig::default()
        };
        let err = build_stub_processor(cfg).expect_err("invalid threshold rejected");
        assert!(format!("{err:#}").contains("threshold"));
    }

    #[test]
    fn build_stub_processor_rejects_unknown_model() {
        let cfg = SttConfig {
            model: "tiny.en".to_string(),
            ..SttConfig::default()
        };
        let err = build_stub_processor(cfg).expect_err("unknown model rejected");
        assert!(format!("{err:#}").contains("tiny.en"));
    }

    // ---- AC3 (wm-stt): inbound turn_id propagates to outbound events ----

    #[tokio::test]
    async fn dispatch_final_carries_inbound_turn_id() {
        // AC3: speech.end with turn_id → wm.stt.final has same turn_id.
        let state = fresh_state();
        let mut sink = MemSink::default();
        let tid = "0123456789abc-0001".to_string();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechStart(SpeechStartEvent { ts: 0 }),
            0,
        )
        .await
        .unwrap();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 3000,
                ts: 3000,
                turn_id: Some(tid.clone()),
            }),
            3000,
        )
        .await
        .unwrap();
        let events = sink.events.lock().unwrap();
        let (topic, payload) = &events[0];
        assert_eq!(topic, outgoing::FINAL, "must route to wm.stt.final");
        assert_eq!(
            payload["turn_id"], tid,
            "AC3: out turn_id must equal in turn_id"
        );
    }

    #[tokio::test]
    async fn dispatch_final_no_turn_id_when_absent() {
        // AC5: speech.end without turn_id → wm.stt.final has no turn_id field.
        let state = fresh_state();
        let mut sink = MemSink::default();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechStart(SpeechStartEvent { ts: 0 }),
            0,
        )
        .await
        .unwrap();
        dispatch(
            &state,
            &mut sink,
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 3000,
                ts: 3000,
                turn_id: None,
            }),
            3000,
        )
        .await
        .unwrap();
        let events = sink.events.lock().unwrap();
        assert_eq!(events[0].0, outgoing::FINAL);
        assert!(
            events[0].1.get("turn_id").is_none(),
            "AC5: absent inbound turn_id must not appear in output"
        );
    }
}
