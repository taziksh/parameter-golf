# Intra-document attention masking vs baseline (1×H100, paired, SEED=42)

Paired comparison on the bigbag SP8192 stack.
`work/intradoc/train_gpt.py` = intra-doc mask on,
`work/intradoc/train_gpt_baseline.py` = off.
Same seed, same 80 train shards, same config.

## Results

| branch    | seed | iters | val_bpb     | val_loss     |
|-----------|-----:|------:|------------:|-------------:|
| baseline  |   42 |  4550 | 1.08077590    | 2.79175639    |
| intradoc  |   42 |  4550 | 1.08091991    | 2.79212840    |

**Δ (intradoc − baseline): +0.0001 val_bpb**

## Setup

- Hardware: 1×H100 80GB
- Branch: `doc-mask` @ e4eea38
- Data: SP8192, **80 train shards**, `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`
- Iterations: 4550, `MAX_WALLCLOCK_SECONDS=0`, `TTT_ENABLED=0`
- Single seed (42) — not enough to cleanly separate sub-σ effects
- Run codes: baseline_rc=0 intradoc_rc=0

## Files

- `baseline.log`, `intradoc.log` — full train logs
- `baseline.final_model.int6.ptz`, `intradoc.final_model.int6.ptz` — quantized artifacts
- `results.tsv` — machine-readable
