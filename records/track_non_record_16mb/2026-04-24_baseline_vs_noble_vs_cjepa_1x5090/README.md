# Non-Record Experiment: Baseline vs NOBLE vs Causal-JEPA (1×RTX 5090)

A small A/B/C sweep to put numbers on two variant tracks — **NOBLE** (LoRA + Adam-for-matrices) and **Causal-JEPA** (auxiliary predictor loss over a teacher-EMA of hidden states) — against the stock `main` baseline, all trained identically on a single RTX 5090.

This is a *non-record* run — hardware and step count are far from the 8×H100 leaderboard regime — but the variants' *relative* behavior is what we care about here.

## Results

Per-seed (`val_bpb` is the scored post-quant metric; lower is better):

| branch   | seed | val_bpb | val_loss | step_avg (ms) | wallclock (s) |
|----------|-----:|--------:|---------:|--------------:|--------------:|
| baseline | 1337 |  1.2608 |   2.1288 |           607 |          2578 |
| baseline | 1338 |  1.2600 |   2.1274 |           605 |          2600 |
| baseline | 1339 |  1.2595 |   2.1266 |           605 |          2598 |
| noble    | 1337 |  1.4033 |   2.3693 |           840 |          3574 |
| noble    | 1338 |  1.4032 |   2.3692 |           839 |          3572 |
| noble    | 1339 |  1.4047 |   2.3719 |           838 |          3570 |
| cjepa    | 1337 |  1.2591 |   2.1260 |           706 |          2981 |
| cjepa    | 1338 |  1.2616 |   2.1302 |           702 |          2980 |
| cjepa    | 1339 |  1.2619 |   2.1306 |           702 |          2959 |

Aggregate (n=3 per branch):

| branch   | mean val_bpb | std (n=3) | Δ vs baseline |
|----------|-------------:|----------:|--------------:|
| baseline |       1.2601 |    0.0007 |             — |
| cjepa    |       1.2609 |    0.0015 |       +0.0008 |
| noble    |       1.4037 |    0.0008 |       +0.1436 |

(Also available as `results.tsv` in this directory.)

## Takeaways

**CJEPA ≈ baseline.** The cjepa−baseline gap (0.0008) is smaller than the cjepa within-branch spread (0.0028). At 4,000 iters the causal-JEPA auxiliary loss adds no measurable win in final val_bpb, and costs ~17% extra wallclock (706 vs 605 ms/step). Also no harm — the representation is fine, the aux loss just isn't buying anything at this budget.

**NOBLE is a clear regression (+0.1436 bpb).** Far outside seed noise — pooled σ is ~0.0008, so the gap is ~180× the pooled std. The tested config is paper-faithful: LoRA rank 32 on the 2D matrices, and Adam on the base matrices instead of Muon (`USE_MUON=0`). Best read as the **Adam-vs-Muon gap dominating** whatever LoRA is or isn't doing. Obvious follow-up: `USE_NOBLE=1 USE_MUON=1` to isolate LoRA from the optimizer choice.

**Step-time cost (vs baseline 605 ms/step):**
- cjepa: +17% (706 ms) — aux loss every step, EMA teacher pass
- noble: +39% (840 ms) — extra LoRA rank-32 path per matrix

## Setup

| | |
|---|---|
| Hardware | 1×NVIDIA RTX 5090 (32 GB, sm_120) |
| Driver / CUDA | 570.124.06 / 12.8 |
| PyTorch | 2.11.0+cu128 |
| Data | FineWeb 10B, SP1024 tokenizer (stock) |
| Iterations | 4,000 per run |
| Seeds | 1337, 1338, 1339 |
| Budget | ~10h for 9 runs (3 branches × 3 seeds) |

### Pinned branch SHAs (this repo)

| branch      | SHA                                        |
|-------------|--------------------------------------------|
| baseline    | `75700cb8d599321e120a16abd13d8382dc97abaf` (main) |
| noble       | `540a3181eaf7e7d908b2435a0ca52ef42e3435b1` |
| causal-jepa | `9bc8b466801ab854637949880a8210b36a6bf7bf` |

### Model/training config (shared, all defaults)

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
TIE_EMBEDDINGS=1 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288
WARMUP_STEPS=20 WARMDOWN_ITERS=1200
```

Variant-specific envs on top of the above:

- **baseline**: none
- **noble**: `USE_NOBLE=1 USE_MUON=0 LORA_RANK=32`
- **cjepa**: none (defaults: `CJEPA_ENABLED=1 CJEPA_K=4 CJEPA_LAMBDA=0.5`)

### Exact launch commands (per run)

```bash
# baseline (from repo root, branch main at the SHA above)
SEED=<1337|1338|1339> ITERATIONS=4000 VAL_LOSS_EVERY=1000 MAX_WALLCLOCK_SECONDS=0 \
  python3 -u train_gpt.py

# noble (branch noble at the SHA above)
USE_NOBLE=1 USE_MUON=0 LORA_RANK=32 \
  SEED=<1337|1338|1339> ITERATIONS=4000 VAL_LOSS_EVERY=1000 MAX_WALLCLOCK_SECONDS=0 \
  python3 -u train_gpt.py

# cjepa (branch causal-jepa at the SHA above)
SEED=<1337|1338|1339> ITERATIONS=4000 VAL_LOSS_EVERY=1000 MAX_WALLCLOCK_SECONDS=0 \
  python3 -u train_gpt_cjepa.py
```

Data/tokenizer paths default to `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/fineweb_1024_bpe.model`. Run `python3 data/cached_challenge_fineweb.py` from repo root if those are missing.

Final metric is the line: `final_int8_zlib_roundtrip val_loss:<x> val_bpb:<y> ...` at the end of the training log.

### Sweep wallclock

- 9 main runs: **27,011 s** (~7.5 h)
  - 3× baseline: 7,776 s (2,578 + 2,600 + 2,598)
  - 3× noble: 10,716 s (3,574 + 3,572 + 3,570)
  - 3× cjepa: 8,920 s (2,981 + 2,980 + 2,959)
- +1 bonus cjepa 8000-iter: 5,848 s (~1.6 h)
- **Full sweep: 33,260 s (9.24 h)** (09:56 → 18:10 UTC on 2026-04-24)

## Methodology Notes

1. **Calibration caveat.** A first pass used `total_wallclock / iters` from a 200-step calibration, which bakes ~140 s of torch.compile + final-eval overhead into per-iter time and inflates it ~2×. The corrected per-iter numbers used to size `ITERATIONS=4000` came from the steady-state `step_avg` at step 200 of the calibration logs (605 / 831 / 722 ms for baseline / noble / cjepa).
2. **Environment fix.** The host shipped with `torch 2.4.1+cu124`, which does not support the 5090's sm_120 (immediate "no kernel image is available for execution" crash on any CUDA kernel). Upgraded to `torch 2.11.0+cu128` before anything trained.
3. **No hyperparameter changes.** Only the envs listed above were set per run; everything else is repo defaults at the pinned SHAs.

## Caveats

- Single 5090, 4,000 iters (~2.1B tokens seen at `TRAIN_BATCH_TOKENS=524288`). Not comparable to 8×H100 / ~5–10k-iter leaderboard entries — neither in absolute score nor in what "converged" means.
- 3 seeds per branch. Enough to cleanly separate noble from baseline (~180σ gap); **not** enough to detect a sub-0.003 bpb cjepa effect if one exists.
- NOBLE tested only in its `USE_MUON=0` form. The Muon+LoRA cell (`USE_NOBLE=1 USE_MUON=1`) is the obvious missing experiment.
- CJEPA tested only at defaults (`K=4`, `λ=0.5`). No sweep.

## Bonus run (double-length cjepa)

One extra `cjepa seed=1340 iters=8000` run to see what happens at 2× the budget:

- **Final `val_bpb` (post-warmdown, int8+zlib): 1.2392**
- val_loss: 2.0923, wallclock: 5,848 s
- Δ vs 4,000-iter cjepa mean (1.2609): **−0.0217** (cleanly better with 2× iters)

Pre-warmdown val trajectory (for shape, not direct comparison to the 4,000-iter final numbers — those had warmdown complete):

| step | val_bpb |
|-----:|--------:|
| 1000 |  1.3839 |
| 2000 |  1.3250 |
| 3000 |  1.3001 |
| 4000 |  1.2853 |
| 5000 |  1.2752 |
| 6000 |  1.2692 |
| 8000 (final, post-warmdown + int8+zlib) | **1.2392** |

Caveat: this is **one seed** and has **no matching baseline@8000** — so it tells you that cjepa keeps improving with more training, but *not* whether cjepa beats a `main` baseline at the same budget.

## TODO / Open follow-ups

- [ ] **Baseline @ 8000 iters, seed 1340** — the one missing cell. Pairs directly with the cjepa bonus above and finally answers "does cjepa actually help, or is cjepa@8k just beating cjepa@4k because of extra training?" Expected wallclock ~84 min on 1×5090 (8000 × 605 ms + ~190 s overhead).
- [ ] **NOBLE + Muon** (`USE_NOBLE=1 USE_MUON=1 LORA_RANK=32`) — isolates the LoRA effect from the Adam-vs-Muon confound that drove the ~0.14 bpb regression seen here.
- [ ] **CJEPA hyperparameter sweep** — `CJEPA_K` and `CJEPA_LAMBDA` were both at defaults (4 and 0.5). Worth at least a 2×2 sweep before concluding the aux loss is a no-op.

## Files

- `README.md` — this file
- `results.tsv` — per-run table, tab-separated, machine-readable
- `run_summary.txt` — raw orchestrator summary (name, iters, seed, wallclock, rc)
