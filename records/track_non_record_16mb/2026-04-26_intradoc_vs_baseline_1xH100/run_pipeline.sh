#!/usr/bin/env bash
# Autonomous pipeline: data → baseline → intradoc → results → commit → push.
# Logs everything to /workspace/parameter-golf/run.log.

set -uo pipefail
exec > >(tee -a /workspace/parameter-golf/run.log) 2>&1

echo "==== pipeline start: $(date -u +%FT%TZ) ===="

export PATH="$HOME/.local/bin:$PATH"
cd /workspace/parameter-golf
source .venv/bin/activate

# ----- Data: ensure all 80 train shards + tokenizer -----
echo "==== data: downloading 80 shards + tokenizer ($(date -u +%FT%TZ)) ===="
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
echo "train_shards_on_disk=$(ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin 2>/dev/null | wc -l)"
echo "val_shards_on_disk=$(ls data/datasets/fineweb10B_sp8192/fineweb_val_*.bin 2>/dev/null | wc -l)"
ls data/tokenizers/ 2>/dev/null || echo "WARN: no data/tokenizers dir"

# ----- Workspace for the runs -----
ln -sfn "$(pwd)/data" work/intradoc/data
mkdir -p work/intradoc/logs
cd work/intradoc

DATE=$(date -u +%Y-%m-%d)
RECORD_DIR="../../records/track_non_record_16mb/${DATE}_intradoc_vs_baseline_1xH100"
mkdir -p "$RECORD_DIR"

# ----- BASELINE -----
echo "==== baseline: start ($(date -u +%FT%TZ)) ===="
DATA_DIR=./data ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=0 SEED=42 \
RUN_ID=baseline VOCAB_SIZE=8192 TTT_ENABLED=0 \
python3 train_gpt_baseline.py 2>&1 | tee logs/baseline.console.txt
BASELINE_RC=${PIPESTATUS[0]}
echo "baseline_rc=${BASELINE_RC}"
cp -v logs/baseline.txt           "$RECORD_DIR/baseline.log" 2>/dev/null || true
cp -v final_model.int6.ptz        "$RECORD_DIR/baseline.final_model.int6.ptz" 2>/dev/null || true
cp -v final_model.pt              final_model.baseline.pt 2>/dev/null || true
echo "==== baseline: end ($(date -u +%FT%TZ)) ===="

# ----- DOC-MASK (paired, same SEED) -----
echo "==== intradoc: start ($(date -u +%FT%TZ)) ===="
DATA_DIR=./data ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=0 SEED=42 \
RUN_ID=intradoc VOCAB_SIZE=8192 TTT_ENABLED=0 \
python3 train_gpt.py 2>&1 | tee logs/intradoc.console.txt
INTRADOC_RC=${PIPESTATUS[0]}
echo "intradoc_rc=${INTRADOC_RC}"
cp -v logs/intradoc.txt           "$RECORD_DIR/intradoc.log" 2>/dev/null || true
cp -v final_model.int6.ptz        "$RECORD_DIR/intradoc.final_model.int6.ptz" 2>/dev/null || true
echo "==== intradoc: end ($(date -u +%FT%TZ)) ===="

# ----- Results aggregation -----
echo "==== results: aggregate + commit ($(date -u +%FT%TZ)) ===="
B_BPB=$(grep -oE 'val_bpb:[0-9.]+'  "$RECORD_DIR/baseline.log" 2>/dev/null | tail -1 | cut -d: -f2)
I_BPB=$(grep -oE 'val_bpb:[0-9.]+'  "$RECORD_DIR/intradoc.log" 2>/dev/null | tail -1 | cut -d: -f2)
B_LOSS=$(grep -oE 'val_loss:[0-9.]+' "$RECORD_DIR/baseline.log" 2>/dev/null | tail -1 | cut -d: -f2)
I_LOSS=$(grep -oE 'val_loss:[0-9.]+' "$RECORD_DIR/intradoc.log" 2>/dev/null | tail -1 | cut -d: -f2)
B_BPB="${B_BPB:-N/A}"; I_BPB="${I_BPB:-N/A}"; B_LOSS="${B_LOSS:-N/A}"; I_LOSS="${I_LOSS:-N/A}"
if [[ "$B_BPB" != "N/A" && "$I_BPB" != "N/A" ]]; then
  DELTA=$(python3 -c "print(f'{${I_BPB} - ${B_BPB}:+.4f}')")
else
  DELTA="N/A"
fi
echo "B_BPB=$B_BPB I_BPB=$I_BPB DELTA=$DELTA"

printf "branch\tseed\titers\tval_bpb\tval_loss\nbaseline\t42\t4550\t%s\t%s\nintradoc\t42\t4550\t%s\t%s\n" \
  "$B_BPB" "$B_LOSS" "$I_BPB" "$I_LOSS" > "$RECORD_DIR/results.tsv"

REPO_SHA=$(git -C ../.. rev-parse --short HEAD)
cat > "$RECORD_DIR/README.md" <<EOF
# Intra-document attention masking vs baseline (1×H100, paired, SEED=42)

Paired comparison on the bigbag SP8192 stack.
\`work/intradoc/train_gpt.py\` = intra-doc mask on,
\`work/intradoc/train_gpt_baseline.py\` = off.
Same seed, same 80 train shards, same config.

## Results

| branch    | seed | iters | val_bpb     | val_loss     |
|-----------|-----:|------:|------------:|-------------:|
| baseline  |   42 |  4550 | ${B_BPB}    | ${B_LOSS}    |
| intradoc  |   42 |  4550 | ${I_BPB}    | ${I_LOSS}    |

**Δ (intradoc − baseline): ${DELTA} val_bpb**

## Setup

- Hardware: 1×H100 80GB
- Branch: \`doc-mask\` @ ${REPO_SHA}
- Data: SP8192, **80 train shards**, \`MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf\`
- Iterations: 4550, \`MAX_WALLCLOCK_SECONDS=0\`, \`TTT_ENABLED=0\`
- Single seed (42) — not enough to cleanly separate sub-σ effects
- Run codes: baseline_rc=${BASELINE_RC} intradoc_rc=${INTRADOC_RC}

## Files

- \`baseline.log\`, \`intradoc.log\` — full train logs
- \`baseline.final_model.int6.ptz\`, \`intradoc.final_model.int6.ptz\` — quantized artifacts
- \`results.tsv\` — machine-readable
EOF

# ----- Commit + push to doc-mask -----
cd ../..
git config user.email "tazikshahjahan@gmail.com"
git config user.name  "Tazik Shahjahan"
git add "records/track_non_record_16mb/${DATE}_intradoc_vs_baseline_1xH100/"
git commit -m "Record: intra-doc mask vs baseline (1×H100, SEED=42, intradoc=${I_BPB} vs baseline=${B_BPB}, Δ=${DELTA})"

git -c "http.extraheader=AUTHORIZATION: bearer ${GH_TOKEN}" \
  push origin doc-mask
PUSH_RC=$?
echo "push_rc=${PUSH_RC}"

echo "==== pipeline done: $(date -u +%FT%TZ) ===="
echo "SUMMARY: baseline=${B_BPB} intradoc=${I_BPB} delta=${DELTA} push_rc=${PUSH_RC}"
