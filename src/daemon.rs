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
//! iter-5 ships the [`StubEngine`] in the production path. The
//! `whisper-rs` engine binding lands in iter-6; the trait boundary at
//! [`crate::engine::TranscriptionEngine`] means only [`run`] needs to
//! change to swap the engine.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::bus::{self, ErrorEvent, Request, decode_request, now_unix_ms, outgoing};
use crate::engine::{StubEngine, TranscriptionEngine};
use crate::processor::{Emit, UtteranceProcessor};
use crate::SttConfig;

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
pub struct AgoraSink {
    pub(crate) inner: agorabus::Client,
}

#[async_trait::async_trait]
impl EventSink for AgoraSink {
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
        let reply = self.inner.publish(topic, data).await?;
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
/// Wraps a single [`UtteranceProcessor`] in a `tokio::sync::Mutex`
/// because the processor mutates on `handle` and is contended by the
/// dispatch loop. Iter-5 only ever has one inflight `dispatch` at a
/// time — barge-in via concurrent requests lands in iter-7 when a
/// separate task races partials.
pub struct DaemonState<E: TranscriptionEngine + Send> {
    /// The state machine that decides which events to publish.
    pub processor: Mutex<UtteranceProcessor<E>>,
}

impl<E: TranscriptionEngine + Send> DaemonState<E> {
    /// Construct a daemon state wrapping an already-built processor.
    #[must_use]
    pub const fn new(processor: UtteranceProcessor<E>) -> Self {
        Self {
            processor: Mutex::const_new(processor),
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
/// # Errors
/// Returns the first publish failure encountered while flushing the
/// processor's emit stream. The outer loop logs and continues.
pub async fn dispatch<E: TranscriptionEngine + Send>(
    state: &DaemonState<E>,
    publish: &mut dyn EventSink,
    req: Request,
    now_ms: u64,
) -> Result<()> {
    let emits = {
        let mut p = state.processor.lock().await;
        p.handle(req, now_ms)
    };
    for emit in emits {
        let topic = topic_for_emit(&emit);
        let payload = emit_to_value(&emit)?;
        publish.publish(topic, payload).await?;
    }
    Ok(())
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

/// Run the live daemon: build the processor, connect to agorabus,
/// subscribe to both inbound prefixes, dispatch each event until the
/// bus closes.
///
/// # Errors
/// Propagates I/O failures from config validation or the agorabus
/// client. A missing agorabus socket is *not* an error: the daemon
/// logs and exits cleanly so the systemd unit restarts it when the bus
/// comes back (same pattern as `wm-tts`).
pub async fn run(cfg: SttConfig) -> Result<()> {
    let processor = build_stub_processor(cfg)?;
    let state = Arc::new(DaemonState::new(processor));

    let sock = agorabus::default_socket_path();
    let Some(mut sub_client) = agorabus::Client::try_connect(&sock).await? else {
        warn!(socket = %sock.display(), "wm-stt: agorabus not reachable; exiting");
        return Ok(());
    };
    sub_client.subscribe(bus::AUDIO_TOPIC_PREFIX).await?;
    sub_client.subscribe(bus::STT_COMMAND_PREFIX).await?;
    info!(
        audio_prefix = bus::AUDIO_TOPIC_PREFIX,
        stt_prefix = bus::STT_COMMAND_PREFIX,
        "wm-stt: subscribed"
    );

    let pub_client = agorabus::Client::connect(&sock).await?;
    let mut sink = AgoraSink { inner: pub_client };

    while let Some(ev) = sub_client.next_event().await? {
        match decode_request(&ev.topic, &ev.data) {
            Ok(req) => {
                let now = now_unix_ms();
                if let Err(err) = dispatch(state.as_ref(), &mut sink, req, now).await {
                    error!(topic = %ev.topic, err = %err, "wm-stt: dispatch failed");
                    let _ = publish_error(&mut sink, "bus", &format!("dispatch: {err}")).await;
                }
            }
            Err(err) => {
                warn!(topic = %ev.topic, err = %err, "wm-stt: decode failed");
                let _ = publish_error(&mut sink, "bus", &format!("decode: {err}")).await;
            }
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
            state.as_ref(),
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
            state.as_ref(),
            &mut sink,
            Request::SpeechStart(SpeechStartEvent { ts: 0 }),
            0,
        )
        .await
        .unwrap();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 5000,
                ts: 5000,
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
            state.as_ref(),
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
            state.as_ref(),
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
            state.as_ref(),
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
            state.as_ref(),
            &mut sink,
            Request::SpeechStart(SpeechStartEvent { ts: 0 }),
            0,
        )
        .await
        .unwrap();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::SpeechEnd(SpeechEndEvent {
                duration_ms: 500,
                ts: 500,
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
            })),
            outgoing::FINAL
        );
        assert_eq!(
            topic_for_emit(&Emit::Uncertain(U {
                text: String::new(),
                confidence: 0.1,
                ts: 0,
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
}
