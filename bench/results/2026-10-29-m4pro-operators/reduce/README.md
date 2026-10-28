# reduce experiment

Base: 4256a5119bd093f6088bd46943aaacf5cc60d48a. Apple M4 Pro, 20 GPU cores, 48 GB, macOS 15.7.1 (24G231), AC power.

Isolated GPU command timings, not kapitsa bench or full-request speedups. A is the unmodified reference kernel; B is the candidate. Six interleaved pairs per shape after warmup; every measured sample is in bench.jsonl. Thermal state is captured with each sample. ChatGPT/Codex remained open as requested.

Small cases used MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 before unrestricted sizes. Model-level checks are recorded separately, if run.
