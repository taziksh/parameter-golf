# Partial results: BASELINE done, intradoc still running

**As of 2026-04-26T17:14:28Z: baseline finished cleanly (rc=0). Intradoc started 17:08:58Z, in progress.**
This file is committed early so the baseline number is safe off-disk.
A full README.md will replace it once both runs complete.

## Baseline (`work/intradoc/train_gpt_baseline.py`)

| eval flavor | val_loss | val_bpb |
|---|---:|---:|
| pre-quantization post-ema (fp32) |  |  |
| quantized (int6+brotli, packed eval) |  |  |
| **quantized_sliding_window (scored)** | **** | **** |

Total submission size: 16,024,779 bytes (int6+brotli artifact 15,976,196 + code 48,583).

## Setup

- Hardware: 1×H100 80GB
- Branch: `doc-mask` @ 8c6fb1b
- Data: SP8192, 80 train shards, `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`
- Iterations: 4550, `MAX_WALLCLOCK_SECONDS=0`, `TTT_ENABLED=0`, SEED=42
- Model: 35,944,536 params, 11 layers, model_dim=512, vocab=8192
- Wallclock: ~95 min training, ~12 min sliding-window eval
