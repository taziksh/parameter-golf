# Setup

Intra-document attention masking on top of the bigbag SP8192 stack
(`records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`).

## Hardware

NVIDIA Hopper. FA3 only ships for Hopper. The 600s wallclock target requires
8×H100 SXM; 1×H100 80GB runs the same schedule in roughly 90 minutes.

## Install

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install \
  https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl
uv pip install numpy sentencepiece huggingface-hub brotli
```

Add `scipy` to run `bench_attn.py`.

## Data

```
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

The default repo has no SP8192 shards; `MATCHED_FINEWEB_REPO_ID` is required.
`--train-shards N` downloads shards `[0, N)` with no random sampling, so two
runs in a comparison must use the same N. Use 80 to match the bigbag record;
use 40 to halve download time when only a paired comparison is needed.

To keep data on a separate volume:

```
mkdir -p /workspace/parameter-golf/datasets
mv data/datasets /workspace/parameter-golf/datasets
ln -s /workspace/parameter-golf/datasets data/datasets
```

## Run

```
DATA_DIR=./data \
ITERATIONS=4550 \
MAX_WALLCLOCK_SECONDS=0 \
SEED=42 \
RUN_ID=intradoc \
VOCAB_SIZE=8192 \
TTT_ENABLED=0 \
python3 train_gpt.py
```

`MAX_WALLCLOCK_SECONDS=0` disables the 600s training cap. `TTT_ENABLED=0`
skips eval-time test-time training.

Outputs:

- `logs/{RUN_ID}.txt` — train log; `val_bpb` is reported here
- `final_model.pt` — fp32 state dict, overwritten each run
- `final_model.int6.ptz` — int6/int8 + brotli artifact

For a paired comparison, run twice with the same `SEED` against
`train_gpt.py` and `train_gpt_baseline.py`.

## Caveats

- Container disk on Runpod is wiped on Stop. Keep work on `/workspace` or
  `git push` first.
- `final_model.pt` is overwritten between runs. Snapshot it elsewhere if
  comparing artifacts after the fact.
- `MAX_SUBDOCS_PER_WINDOW = 8` covers 99% of SP8192 windows. Raising it
  costs attention step time; lowering merges surplus BOSes into the last
  sub-document of rare large-doc-count windows.
