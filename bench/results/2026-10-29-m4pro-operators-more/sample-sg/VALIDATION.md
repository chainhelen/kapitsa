## What changed and why

Qwen35's ordinary sampled decode uses the existing `topk_stage1_rows_sg` for K >= 8. The candidate lists keep the same order; greedy and smaller K retain `topk_stage1`. Stage 2, RNG, and speculative-verifier dispatches are unchanged.

## Measured

Stage 1 over a 248,320-token vocabulary, K=32, one row. This is selection time, not complete token latency. In the final repeat, median isolated, warm-buffer GPU command-buffer time is 46.232 → 36.325 µs (21.4% lower). The test prints all six interleaved A/B pairs for every shape; initial experiments and repeats are retained.

Runs: Apple M4 Pro, 20 GPU cores, 48 GB, macOS 15.7.1 (24G231), AC power, Qwen3.8-27B-4bit. Main `4256a51`, native candidate `7062e9d`. Kapitsa 0.2.24. Flags: default Metal arithmetic, unchanged numerical floors.

| Pair (A then B) | Main µs | Candidate µs | Thermal A / B |
|---|---:|---:|---|
| 1 | 46.257 | 35.676 | nominal / nominal |
| 2 | 46.285 | 36.333 | nominal / nominal |
| 3 | 46.124 | 36.317 | nominal / nominal |
| 4 | 46.237 | 35.884 | nominal / nominal |
| 5 | 45.744 | 36.337 | nominal / nominal |
| 6 | 46.228 | 36.578 | nominal / nominal |

Unmodified `kapitsa bench --quick --json` produced the following default DFlash2 runs (fresh `KAPITSA_HOME` per leg). The operator measurements do not establish a whole-request speedup. All original JSONs include individual samples and conditions; ChatGPT remained running. The default drafter is `/Users/bytedance/.cache/kapitsa/models/qwen3.8-27b-4bit/dflash2`, weights SHA-256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c` (the exact checksum is verified in the committed run evidence).

| Run | T=0.6 tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Thermal | Other GPU mean % |
|---|---:|---:|---:|---:|---|---:|
| A1 | 59.69 | 65.37 | 10.4748 | 0.2466 | nominal | 0.0 |
| B1 | 60.06 | 66.03 | 10.4638 | 0.2456 | nominal | 0.0 |
| A2 | 55.55 | 64.81 | 10.4785 | 0.2444 | fair | 0.0 |
| B2 | 62.2 | 67.05 | 10.4741 | 0.2373 | fair | 0.0 |

## Sampled single-row decoding

`tools/bench_sample_decode.py` uses the actual HTTP server with `--spec 0`, temperature 0.7, top-k 32 and seed 42. It waits for the ANE background program before warmup. All four outputs are nonempty, reach 512 tokens, have identical SHA-256, and report drafter source `none`. This explicitly engages the changed sampling dispatch; the default speculative benchmark alone does not.

| Run | Output tokens | Decode tok/s | Prefill ms | Thermal before → after |
|---|---:|---:|---:|---|
| A1 | 512 | 15.578 | 7714.0 | 1 → 1 |
| B1 | 512 | 15.600 | 7700.9 | 1 → 1 |
| A2 | 512 | 15.592 | 7700.7 | 1 → 1 |
| B2 | 512 | 15.614 | 7699.4 | 1 → 1 |

## Checked

- `sample_reduce_test --gpu --small` under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, then `--gpu`: 360 cases, zero failures. Checks include the original kernel, independent CPU reference, tails/ties where applicable, and output guards.
- Default 27B output gate: all 584 rows bitwise identical to main (8 prefill + 192 step + 192 verify + 192 batch). Floors were not changed.
- All 27 CPU check groups, the installed-model tokenizer grammar/mask checks, and `prek run --all-files` passed. Named targets were built on the macOS 15 SDK.
- API: 48/49, with the same keepalive-fixture failure reproduced on main in the earlier control. Its holder prompts end after two tokens, so the waiter receives a token before the expected 15-second keepalive.

Evidence: `bench/results/2026-10-29-m4pro-operators-more/sample-sg/`.

<details>
<summary>Default logit comparison</summary>

```text
prefill: 8 records (qwen3.8-27b-4bit)
  reference: 0.2.24 (4256a51), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.24 (7062e9d), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
batch: 192 records (qwen3.8-27b-4bit)
  reference: 0.2.24 (4256a51), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.24 (7062e9d), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
  PASS (../kapitsa-perf-sample-sg/tools/logit_floors/qwen3.8-27b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
step: 192 records (qwen3.8-27b-4bit)
  reference: 0.2.24 (4256a51), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.24 (7062e9d), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
  PASS (../kapitsa-perf-sample-sg/tools/logit_floors/qwen3.8-27b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
verify: 192 records (qwen3.8-27b-4bit)
  reference: 0.2.24 (4256a51), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.24 (7062e9d), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
  PASS (../kapitsa-perf-sample-sg/tools/logit_floors/qwen3.8-27b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
```
</details>

## Not verified

Qwen3.5-4B and Qwen3.6-35B-A3B/MoE (not installed), other Apple chips/macOS versions, Metal-4-only tools, and real image end-to-end requests. Measured gains are for the stated operator shapes and paths, not a general request-speed claim.
