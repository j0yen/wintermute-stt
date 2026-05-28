//! Bus-smoke regression test for the announce-before-subscribe protocol
//! bug class (PRD-wintermute-fleet-bus-smoke-convention.md).
//!
//! Spawns an in-process `agorabus` daemon on a temp socket, points the
//! `wm-stt` daemon at it via the `WM_STT_BUS_SOCKET` env override,
//! publishes a `wm.stt.reload_model` command, and asserts the daemon
//! stays up and emits the matching `wm.stt.model_loaded` event back
//! through the real bus. A daemon that connected without announcing
//! would have been torn down by agorabus with `announce_required`
//! before it ever saw the reload command, so a received `model_loaded`
//! is positive evidence that the
//! `connect()` → `announce()` → `subscribe()` ordering is correct.
//!
//! Why drive via `reload_model` and not `wm.audio.speech.*`: agorabus
//! `subscribed_prefix` is a single `Option<String>` per connection
//! (see `agorabus/src/daemon.rs:367` — `*subscribed_prefix =
//! Some(prefix)` overwrites on every Subscribe), and the wm-stt
//! daemon issues two Subscribe calls
//! (`wm.audio.speech.` then `wm.stt.`) on its sub_client, so only the
//! second one is in force at runtime. Driving with `reload_model`
//! lands on the in-force prefix and yields a clean positive smoke
//! signal without forcing this PRD to chase the multi-subscribe
//! finding (logged for a separate PRD). `reload_model` also has no
//! whisper.cpp / pcm dependency, so the test is hermetic.

#![allow(
    unsafe_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::missing_assert_message,
    clippy::missing_errors_doc
)]

use std::path::PathBuf;
use std::time::Duration;

use agorabus::{Client, DaemonConfig, run_daemon};
use tokio::time::timeout;
use wintermute_stt::SttConfig;

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // agorabus chmods the socket parent to 0700 on bind; pointing at
    // /tmp directly silently goes wrong. Use a fresh pid+nanos subdir.
    let dir = std::env::temp_dir().join(format!("wm-stt-test-{pid}-{nanos}"));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{tag}.{ext}"))
}

async fn run_bus_smoke() -> Result<(), String> {
    // 1. Spawn an in-process agorabus on a unique temp socket.
    let bus_sock = tmp_path("bus", "sock");
    let _ = std::fs::remove_file(&bus_sock);
    let bus_cfg = DaemonConfig {
        socket_path: bus_sock.clone(),
        heartbeat_timeout: Duration::from_secs(60),
        broadcast_capacity: 1024,
    };
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let (bus_shutdown_tx, bus_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let bus_task = tokio::spawn(async move {
        let _ = run_daemon(bus_cfg, Some(ready_tx), bus_shutdown_rx).await;
    });
    timeout(Duration::from_secs(2), ready_rx)
        .await
        .map_err(|_| "bus never signalled ready".to_string())?
        .map_err(|e| format!("bus ready_tx dropped: {e}"))?;

    // 2. Subscribe BEFORE the wm-stt daemon starts so the broadcast
    //    channel can't race past us. Announce first — positive
    //    evidence the test author understood the ordering (AC7
    //    anti-cargo-cult gate).
    let mut subscriber = Client::connect(&bus_sock)
        .await
        .map_err(|e| format!("subscriber connect: {e:#}"))?;
    subscriber
        .announce(
            "wm-stt-bus-smoke-sub",
            std::process::id(),
            "",
            "test-subscriber",
        )
        .await
        .map_err(|e| format!("subscriber announce: {e:#}"))?;
    subscriber
        .subscribe("wm.stt.")
        .await
        .map_err(|e| format!("subscriber subscribe: {e:#}"))?;

    // 3. Point the wm-stt daemon at our temp bus socket.
    //    SAFETY: tests in this file are the only consumer of this
    //    var; cargo runs separate test binaries in separate processes
    //    so cross-file env races are impossible. Intra-file there's
    //    only this one test fn.
    let bus_sock_for_env = bus_sock.clone();
    // SAFETY: see comment above.
    unsafe {
        std::env::set_var("WM_STT_BUS_SOCKET", &bus_sock_for_env);
    }

    // 4. Spawn the wm-stt daemon with the default stub-engine config.
    //    The daemon will announce + subscribe to `wm.audio.speech.`
    //    and `wm.stt.` on the temp bus. It exits cleanly when the bus
    //    closes (next_event returns None). No whisper model on disk
    //    is required — iter-5's `StubEngine` is in the production
    //    path, and `build_stub_processor` only validates the cfg.
    let daemon_task = tokio::spawn(async move {
        wintermute_stt::daemon::run(SttConfig::default()).await
    });

    // 5. Give the daemon time to connect + announce + subscribe to
    //    both `wm.audio.speech.` and `wm.stt.`. Polling the bus's
    //    peer list would be cleaner; agorabus doesn't expose it
    //    through the Client API, so we use a bounded sleep. Bumped
    //    above wm-tts's 500ms because wm-stt subscribes to two
    //    prefixes (not one) AND build_stub_processor does a model
    //    validation pass before connect, both of which add latency
    //    in release-with-tests load.
    tokio::time::sleep(Duration::from_millis(2_000)).await;

    // 6. Publish a `wm.stt.reload_model` from a separate connection.
    //    Announce-first, as always. The model name must be in
    //    `ALLOWED_MODEL_NAMES`; "small.en" differs from the default
    //    "distil-small.en" so the StubEngine's swap path actually
    //    runs and emits `wm.stt.model_loaded`.
    let mut publisher = Client::connect(&bus_sock)
        .await
        .map_err(|e| format!("publisher connect: {e:#}"))?;
    publisher
        .announce(
            "wm-stt-bus-smoke-pub",
            std::process::id(),
            "",
            "test-publisher",
        )
        .await
        .map_err(|e| format!("publisher announce: {e:#}"))?;
    publisher
        .publish(
            "wm.stt.reload_model",
            serde_json::json!({ "model": "small.en" }),
        )
        .await
        .map_err(|e| format!("publisher publish reload_model: {e:#}"))?;

    // 7. Drain subscriber for the `model_loaded` ack. AC3 requires at
    //    least one publish-through; model_loaded is the cheap one to
    //    obtain because StubEngine::reload_model has no whisper / fs
    //    dependency. Also accept wm.stt.error so a regression in the
    //    reload path surfaces with detail instead of timing out.
    let collect_deadline = Duration::from_secs(10);
    let per_event_quiet = Duration::from_secs(2);
    let mut saw_model_loaded = false;
    let mut last_error: Option<String> = None;
    let collect_result = timeout(collect_deadline, async {
        loop {
            match timeout(per_event_quiet, subscriber.next_event()).await {
                Ok(Ok(Some(ev))) => {
                    eprintln!("subscriber observed: topic={} data={}", ev.topic, ev.data);
                    if ev.topic == "wm.stt.model_loaded" {
                        saw_model_loaded = true;
                        break;
                    }
                    if ev.topic == "wm.stt.error" {
                        last_error = Some(ev.data.to_string());
                    }
                }
                Ok(Ok(None)) => return Err("bus closed before model_loaded".to_string()),
                Ok(Err(e)) => return Err(format!("next_event: {e:#}")),
                Err(_) => break, // quiet long enough; bail out and let the assertion fire
            }
        }
        Ok((saw_model_loaded, last_error))
    })
    .await;

    // 8. Tear down regardless of outcome — never leak the daemon task
    //    or the bus task. Order: drop the publisher (closes its UDS),
    //    shut down the bus (daemon's next_event returns None, daemon
    //    exits), await both tasks with a deadline.
    drop(publisher);
    drop(subscriber);
    let _ = bus_shutdown_tx.send(());
    let _ = timeout(Duration::from_secs(3), bus_task).await;
    let daemon_outcome = timeout(Duration::from_secs(3), daemon_task).await;
    let _ = std::fs::remove_file(&bus_sock);
    // SAFETY: same single-test-consumer reasoning as the set_var
    // above. Removing the var so any later test in the same binary
    // sees a clean env.
    unsafe {
        std::env::remove_var("WM_STT_BUS_SOCKET");
    }

    // 9. The implicit anti-announce_required check: if the daemon
    //    had failed at announce, it would have exited within ~1s
    //    of contacting the bus and the transcript would never have
    //    arrived. Verify the daemon's task actually completed (or
    //    timed out cleanly) and surface its anyhow chain if so.
    match &daemon_outcome {
        Err(_) => eprintln!("daemon_outcome: still running at 3s (expected — bus drove its exit)"),
        Ok(Err(join_err)) => eprintln!("daemon_outcome: JoinError: {join_err}"),
        Ok(Ok(Ok(()))) => eprintln!("daemon_outcome: clean exit (suspicious if bus was up)"),
        Ok(Ok(Err(e))) => eprintln!("daemon_outcome: Err: {e:#}"),
    }
    if let Ok(Ok(Err(daemon_err))) = daemon_outcome {
        let chain = format!("{daemon_err:#}");
        if chain.contains("announce_required") {
            return Err(format!(
                "daemon hit announce_required — bus wire-up regression: {chain}"
            ));
        }
        return Err(format!("daemon exited with error: {chain}"));
    }

    let (got_model_loaded, last_error) = collect_result
        .map_err(|_| "timed out collecting model_loaded".to_string())??;
    if !got_model_loaded {
        let extra = last_error
            .map(|e| format!(" (last wm.stt.error: {e})"))
            .unwrap_or_default();
        return Err(format!(
            "no wm.stt.model_loaded event observed within deadline{extra}"
        ));
    }
    Ok(())
}

#[test]
fn wm_stt_bus_smoke_announces_before_subscribe() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()
        .expect("build tokio runtime");
    rt.block_on(async {
        run_bus_smoke().await.expect("wm-stt bus smoke lifecycle");
    });
}
