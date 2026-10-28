# Validation

Native code tested: 78a12b2, against main 4256a5119bd093f6088bd46943aaacf5cc60d48a. Later evidence/tool-only changes do not alter src/ or CMake.

Runs: Apple M4 Pro (20 GPU cores), 48 GB, macOS 15.7.1 (24G231), Qwen3.8-27B-4bit, AC power. Default Metal math; unchanged numerical floor.

`rope_pair_test --gpu --small` under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, then `--gpu`: all 144 full-size reference/guard cases passed, covering ordinary and interleaved M-RoPE, non-rotary coordinates and untouched cache rows.

Default output check: all 584 rows bitwise identical. Synthetic 64K stress check: all 17 rows bitwise identical. `logit-compare.txt` and `long-compare.txt` contain the complete reports. The 64K input repeats the first default corpus window's token IDs; it is a correctness stress case, not a throughput workload.

All 27 CPU check groups completed. The first CLI run failed the APFS last-clone space-report assertion; the full CLI retry and the main control passed. Both the failure and retries are retained. The installed model tokenizer's grammar/mask test and final prek also passed.

API: candidate and main both passed 48/49. The same keepalive fixture fails: holder requests stop after two tokens, freeing slots before its expected 15-second keepalive. Logs are included; no test was weakened.

## Standard request benchmark

Produced by unmodified `kapitsa bench --quick --json`, A/B/A/B plus one extra A/B pair, with a fresh KAPITSA_HOME per run. The benchmark filters NI_DFLASH/NI_NO_MTP, so all these runs use default KFlash/DFlash2. Every JSON includes all individual samples, hardware, version, flags and conditions. All were valid, tools passed, swapouts were zero, and contention lists were empty. ChatGPT was retained; its detected GPU activity is shown. Thermal state varies between nominal and fair; no reproducible whole-request speedup is claimed.

| Run | Decode T=0.6 tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Thermal | Other GPU mean % |
|---|---:|---:|---:|---:|---|---:|
| A1 | 64.59 | 64.37 | 10.7198 | 0.241 | nominal | 3.14 |
| B1 | 59.82 | 63.85 | 10.7046 | 0.2309 | fair | 2.2 |
| A2 | 59.02 | 64.6 | 11.0208 | 0.2292 | fair | 2.02 |
| B2 | 59.3 | 65.61 | 10.5065 | 0.2321 | fair | 0.0 |
| A3 | 59.48 | 65.73 | 10.5437 | 0.2293 | fair | 0.0 |
| B3 | 60.15 | 65.76 | 10.5016 | 0.2338 | nominal | 0.02 |


## Not verified

Qwen3.5-4B and Qwen3.6-35B-A3B/MoE (not installed), other chips/macOS versions, and real multimodal end-to-end inputs. Named build targets were used as required on the older SDK; M5/Metal-4-only tools were not built.
