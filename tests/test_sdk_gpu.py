"""Authorized real-model test: python3 tests/test_sdk_gpu.py MODEL_DIR (idle GPU required)."""
import fcntl
import json
import os
import signal
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk/python"))
from kapitsa_sdk import Engine, KapitsaError  # noqa: E402

# Terminate the test process on a deadlock; never free a request still running in native code.
signal.alarm(300)
with tempfile.TemporaryDirectory(prefix="kapitsa-sdk-") as tmp:
    os.environ["BENCH_GPU_LOCK"] = str(Path(tmp) / "gpu.lock")
    for name in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD"):
        os.environ.pop(name, None)
    with ExitStack() as handles:
        engine = handles.enter_context(Engine(sys.argv[1], slots=2, context=4096,
                                              prefill_chunk=512, spec_depth=0, dflash="none"))
        options = dict(max_tokens=16, temperature=0, template_kwargs={"enable_thinking": False})
        messages = [{"role": "user", "content": "Reply with a short greeting."}]
        previous = None
        for label in ("first", "repeat", "callback-cancel"):
            deltas = []
            def delta(value):
                deltas.append(value)
                return label != "callback-cancel"
            result = engine.generate(messages, on_delta=delta, **options)
            assert deltas and result["completion_tokens"] > 0, result
            if label == "callback-cancel":
                assert result["finish_reason"] == "cancelled", result
            else:
                assert result["finish_reason"] != "cancelled", result
                if previous is not None:
                    assert result["content"] == previous, result
                previous = result["content"]
            print(json.dumps({"case": label, "result": result}), flush=True)

        with engine.request(messages, **options) as request:
            request.cancel()
            calls = []
            result = request.generate(on_delta=calls.append)
            assert result["finish_reason"] == "cancelled" and not calls, result
            try:
                request.generate()
            except RuntimeError:
                pass
            else:
                raise AssertionError("a request cannot generate twice")
        print("PASS pre-cancel and one-shot request", flush=True)

        failure = ValueError("application callback failed")
        def fail_callback(_delta):
            raise failure
        try:
            engine.generate(messages, on_delta=fail_callback, **options)
        except ValueError as caught:
            assert caught is failure
        else:
            raise AssertionError("callback exception was swallowed")
        assert engine.generate(messages, **options)["completion_tokens"] > 0
        print("PASS callback exception and subsequent generation", flush=True)

        # Repeatedly reuse both native slots; the last round also releases the public engine handle.
        for round_number in range(3):
            with ExitStack() as request_handles:
                requests = [request_handles.enter_context(engine.request(
                    [{"role": "user", "content": f"List many facts about the number {i + 7 + round_number}."}],
                    **{**options, "max_tokens": 64})) for i in range(2)]
                if round_number == 2:
                    engine.close()
                    try:
                        Engine(sys.argv[1])
                    except KapitsaError as error:
                        assert error.status == 2, error
                    else:
                        raise AssertionError("retained requests must keep the engine alive")
                streamed = [threading.Event(), threading.Event()]
                start = threading.Barrier(3, timeout=30)
                def generate(i):
                    start.wait()
                    return requests[i].generate(on_delta=lambda _delta: streamed[i].set())
                def cancel():
                    start.wait()
                    try:
                        assert all(event.wait(120) for event in streamed), "both requests must stream"
                    finally:
                        requests[0].cancel()
                with ThreadPoolExecutor(max_workers=3) as pool:
                    work = [pool.submit(generate, i) for i in range(2)]
                    pool.submit(cancel).result()
                    results = [future.result() for future in work]
                assert results[0]["finish_reason"] == "cancelled", results
                assert results[1]["finish_reason"] in ("stop", "length"), results
                assert all(result["completion_tokens"] > 0 for result in results), results
                print(json.dumps({"case": "two-requests-third-thread-cancel",
                                  "round": round_number, "results": results}), flush=True)
    # No engine/request handles left: another user can now acquire the lock.
    with open(os.environ["BENCH_GPU_LOCK"]) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
signal.alarm(0)
print("PASS", flush=True)
