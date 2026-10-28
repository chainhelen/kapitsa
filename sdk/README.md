# Kapitsa in-process SDK (prototype)

C ABI with Rust and Python wrappers around `ni::Engine`. Create one engine and reuse
it across requests. Models must already exist locally; no HTTP server, downloads or child processes.

## Build and use

Apple Silicon, macOS 14+, Xcode Command Line Tools, CMake 3.24+; Rust for the Rust wrapper:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_BUILD_SDK=ON
cmake --build build -j 6 --target kapitsa_sdk sdk_contract_test
build/sdk_contract_test
export KAPITSA_LIB_DIR="$(pwd)/build"
cargo test --manifest-path sdk/rust/Cargo.toml
cargo clippy --manifest-path sdk/rust/Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path sdk/rust/Cargo.toml --example chat -- /local/model-directory
```

See [the Rust example](rust/examples/chat.rs) for model reuse and streaming. Engine clones
share one model; cancellation handles can be sent to another thread. Callbacks run
synchronously; panics are caught at the Rust/C boundary.

## Interface and ownership

The [C header](c/kapitsa.h) defines the eight exported functions, options, request fields,
statuses and ownership. Requests retain an engine after its handle is released; destroy
requests only after generation returns. Free result/error strings with `kapitsa_string_free`.

Loading is synchronous. A busy GPU lock returns BUSY immediately; idle engines and retained
requests hold the lock. A host that already owns it uses `NI_LOCK_HELD=1`.
Do not mix separate users of Kapitsa's global GPU runtime in one process.

Upstream `fatal()` calls `_exit(1)` for some loader/Metal/resource failures. These can still
terminate the host despite the SDK's C++ exception handling. Hosts requiring crash isolation
should load the SDK inside their own Worker process.

## Python

Build the native library first, then install the stdlib `ctypes` wrapper:

```sh
python3 -m pip install ./sdk/python
export KAPITSA_LIBRARY="$(pwd)/build/libkapitsa_sdk.dylib"
```

```python
from kapitsa_sdk import Engine
with Engine("/local/model-directory", slots=2) as engine:
    result = engine.generate(
        [{"role": "user", "content": "Hello"}], max_tokens=64,
        on_delta=lambda delta: print(delta),
    )
```

`on_delta` receives a dict; returning `False` cancels. For cross-thread cancellation, create
`engine.request(...)`, run `request.generate()` on a worker and call `request.cancel()` elsewhere.
`ctypes.CDLL` releases the GIL; callbacks run on the generation thread. Callback exceptions
cancel generation and are re-raised after native return. Use context managers or close explicitly;
closing a running request is refused. Requests retain the model after `Engine.close()`.
The wheel contains no native library: set `KAPITSA_LIBRARY` to the built dylib.

CPU tests: `PYTHONPATH=sdk/python python3 -m unittest discover -s sdk/python`.
Authorized real-model checks: `python3 tests/test_sdk_gpu.py MODEL_DIR` (idle GPU required;
uses a temporary GPU lock). See [CONTRIBUTING.md](../CONTRIBUTING.md#sdk-build-and-tests).

## Request and result contract

Omitted sampling fields use the loaded model's defaults. The SDK rejects unknown top-level
fields; its JSON is a library interface, not an OpenAI HTTP request body.

| Engine option | Meaning |
| --- | --- |
| `model_dir` | Required local model folder containing `config.json` and the model files. |
| `context` | Token capacity per slot; default 32768, subject to the engine's memory plan. |
| `slots` | Concurrent conversation slots, 1–4; SDK default 1. More slots need more memory. |
| `prefill_chunk` | Rows per prefill pass; 0 asks the engine to choose. |
| `spec_depth` | -1 uses the model default; 0 disables speculative decoding. |
| `mem_budget_gb` | GPU memory budget in GB; 0 uses the native default. |
| `dflash` | Native drafter selection: `auto`, `none`, or a local directory; empty inherits native configuration. |
| `disk_cache` | SSD prefix-cache budget such as `16G`, `auto`, or `0`; empty inherits `NI_DISK_CACHE`, otherwise off. |

Requests accept `messages`, `tools`, `template_kwargs`, `max_tokens`, `temperature`, `top_p`,
`top_k`, `seed` and `stop`. `messages` is a nonempty array of messages with a string `role`.
Image content uses the existing engine's message handling. Structured-output schemas and model
switching are not exposed by ABI version 1.

A streaming callback receives `content`, `reasoning` and complete `tool_calls` deltas.
The final result contains the accumulated text/tool calls, `finish_reason`, `error`,
`prompt_tokens`, `cached_tokens`, `completion_tokens`, `prefill_ms` and `decode_ms`.
Each tool call has `id`, `name` and `arguments`; `arguments` is JSON **text**, not a parsed object.
`finish_reason` is `stop`, `length`, `tool_calls`, `cancelled` or `error`.

C results and errors are separately allocated strings. Free every non-null output, even when
status is an error. A request cancelled before generation starts has no result string; cancellation
after output begins can return a partial result. Invalid UTF-8 bytes are replaced with U+FFFD.
Consequently concatenated deltas need not equal final text when a character is split across deltas;
use the final result as the complete response. No byte-perfect streaming claim is made.

Cancellation is cooperative: it sets a flag and returns immediately. Join/wait for generation
before destroying the request. The same request cannot generate twice. A cancellation racing
with a completed response can leave that response successful. C handles cannot be destroyed
concurrently with another operation on the same handle; Rust ownership and Python close guards
cover these lifetime rules in their wrappers.

## Rust in another application

Add a path dependency and set `KAPITSA_LIB_DIR` when building the application:

```toml
[dependencies]
kapitsa-sdk = { path = "/path/to/kapitsa/sdk/rust" }
```

```rust
use kapitsa_sdk::Engine;

fn main() -> Result<(), kapitsa_sdk::Error> {
    let engine = Engine::new(r#"{"model_dir":"/local/model-directory","slots":2}"#)?;
    let request = engine.request(r#"{
        "messages":[{"role":"user","content":"Explain Attention briefly"}],
        "max_tokens":128,"temperature":0
    }"#)?;
    let cancel = request.cancellation(); // May move to a controlling thread.
    let output = request.generate(|delta_json| {
        println!("{delta_json}");
        true
    })?;
    println!("{}", output.json);
    drop(cancel); // Cancellation handles also retain the request/model.
    Ok(())
}
```

## Validation

CPU contracts are run in CI without loading a model. Python tests use a scripted native boundary
for callback exceptions and close/cancel races; they do not establish native GPU thread safety.
`tests/test_sdk_gpu.py` covers the real boundary, including repeated two-slot generation,
cross-thread cancellation, callback failure, one-shot requests and retained engine state.

The Rust model test is deliberately ignored by normal `cargo test`. Run it only with an authorized
idle GPU, a local model and a temporary `BENCH_GPU_LOCK` path:

```sh
export KAPITSA_TEST_OPTIONS='{"model_dir":"/local/model-directory","context":4096,"spec_depth":0,"dflash":"none"}'
export BENCH_GPU_LOCK="$(mktemp -t kapitsa-sdk-rust)"
cargo test --manifest-path sdk/rust/Cargo.toml --test model -- --ignored --nocapture
```

It checks callback cancellation, callback panic containment, cancellation from another thread,
and requests retaining a released engine. The engine acquires the lock itself; do not set
`NI_LOCK_HELD` unless a surrounding process really holds it. These are functional checks,
not performance benchmarks or exhaustive concurrency stress. Actual hardware, commit and run
results belong in the PR's `Measured`/`Checked` sections; untested cases belong in `Not verified`.
