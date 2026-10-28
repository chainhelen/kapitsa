//! Opt-in real-model checks. See sdk/README.md; never part of the CPU-only CI run.
use kapitsa_sdk::Engine;
use std::sync::mpsc;
use std::time::Duration;

const REQUEST: &str = r#"{
    "messages":[{"role":"user","content":"Count from one to one hundred."}],
    "max_tokens":32,"temperature":0,"template_kwargs":{"enable_thinking":false}
}"#;

#[test]
#[ignore = "requires a local model and an authorized idle GPU"]
fn model_lifecycle_and_callback_errors() -> Result<(), Box<dyn std::error::Error>> {
    let options = std::env::var("KAPITSA_TEST_OPTIONS")?;
    // The launcher supplies a temporary lock path; Engine acquires and releases it.
    assert!(std::env::var_os("BENCH_GPU_LOCK").is_some());
    let engine = Engine::new(&options)?;

    let request = engine.request(REQUEST)?;
    let cancel = request.cancellation();
    cancel.cancel();
    let output = request.generate(|_| panic!("pre-cancelled request must not stream"))?;
    assert!(output.cancelled);
    drop(cancel);

    let mut callbacks = 0;
    let output = engine.request(REQUEST)?.generate(|_| {
        callbacks += 1;
        false
    })?;
    assert!(callbacks > 0 && output.cancelled);

    // A panic must become an SDK error after native generation stops, not cross the C ABI.
    let error = engine
        .request(REQUEST)?
        .generate(|_| panic!("intentional callback failure"))
        .expect_err("the wrapper must report a callback panic");
    assert_eq!(error.status, 4);
    assert!(error.message.contains("callback panicked"));

    let request = engine.request(REQUEST)?;
    let cancel = request.cancellation();
    let (streamed, started) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        request.generate(|_| {
            let _ = streamed.send(());
            true
        })
    });
    // Stop only after an actual delta; no sleeps that assume a model's speed.
    let ready = started.recv_timeout(Duration::from_secs(60));
    cancel.cancel();
    let output = worker.join().expect("generation thread panicked")?;
    ready?;
    assert!(output.cancelled);
    drop(cancel);

    let request = engine.request(REQUEST)?;
    drop(engine);
    let mut callbacks = 0;
    let output = request.generate(|_| {
        callbacks += 1;
        true
    })?;
    assert!(!output.cancelled && callbacks > 0 && !output.json.is_empty());
    println!(
        "PASS Rust pre-cancel, callback cancellation/panic, cross-thread cancel and retained model"
    );
    Ok(())
}
