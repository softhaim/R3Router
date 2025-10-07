#!/usr/bin/env bash
set -euo pipefail

# ============================================
# R3Route – Step0 (data generation) runner
# --------------------------------------------
# Usage:
#   ./run_step0.sh [csqa|gsm8k|both] [PY=step0_CSQA.py]
# Env overrides (examples):
#   CSQA_TOTAL=5000 CSQA_SEED=800 ./run_step0.sh csqa
#   GSM_TOTAL=6000  GSM_SEED=1000 ./run_step0.sh gsm8k
# ============================================

export PYTHONUNBUFFERED=1

# Positional args (with env fallbacks)
TASK="${1:-${TASK:-csqa}}"
PY="${2:-${PY:-step0_CSQA.py}}"

# ---------- CSQA ----------
CSQA_TOTAL="${CSQA_TOTAL:-4000}"
CSQA_SEED="${CSQA_SEED:-500}"
OUT_CSQA="${OUT_CSQA:-runs/csqa_train.jsonl}"
BANK_CSQA="${BANK_CSQA:-banks/bank_csqa.jsonl}"

# ---------- GSM8K ----------
GSM_TOTAL="${GSM_TOTAL:-4000}"
GSM_SEED="${GSM_SEED:-600}"
OUT_GSM="${OUT_GSM:-runs/gsm8k_train.jsonl}"
BANK_GSM="${BANK_GSM:-banks/bank_gsm8k.jsonl}"

# ---------- GPU ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

# Dirs
mkdir -p \
  "$(dirname "$OUT_CSQA")" \
  "$(dirname "$OUT_GSM")" \
  "$(dirname "$BANK_CSQA")" \
  "$(dirname "$BANK_GSM")"

echo "========== Step0 config =========="
echo "PY=$PY"
echo "TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[CSQA] TOTAL=$CSQA_TOTAL, SEED=$CSQA_SEED, OUT=$OUT_CSQA, BANK=$BANK_CSQA"
echo "[GSM8K] TOTAL=$GSM_TOTAL, SEED=$GSM_SEED, OUT=$OUT_GSM, BANK=$BANK_GSM"
echo "==================================="

run_csqa() {
  local remain=$(( CSQA_TOTAL - CSQA_SEED ))

  echo "[CSQA] Phase 1/2 — seed bank ($CSQA_SEED)"
  python "$PY" \
    --task csqa \
    --csqa_offset 0 \
    --num_samples "$CSQA_SEED" \
    --out "$OUT_CSQA" \
    --bank_dump "$BANK_CSQA"

  if (( remain > 0 )); then
    echo "[CSQA] Phase 2/2 — main with retrieval (offset=$CSQA_SEED, num=$remain)"
    python "$PY" \
      --task csqa \
      --csqa_offset "$CSQA_SEED" \
      --num_samples "$remain" \
      --out "$OUT_CSQA" \
      --ret_bank "$BANK_CSQA" \
      --bank_dump "$BANK_CSQA"
  else
    echo "[CSQA] Phase 2/2 skipped (TOTAL <= SEED)"
  fi
}

run_gsm8k() {
  local remain=$(( GSM_TOTAL - GSM_SEED ))

  echo "[GSM8K] Phase 1/2 — seed bank ($GSM_SEED)"
  python "$PY" \
    --task gsm8k \
    --num_samples "$GSM_SEED" \
    --out "$OUT_GSM" \
    --bank_dump "$BANK_GSM"

  if (( remain > 0 )); then
    echo "[GSM8K] Phase 2/2 — main with retrieval (num=$remain)"
    python "$PY" \
      --task gsm8k \
      --num_samples "$remain" \
      --out "$OUT_GSM" \
      --ret_bank "$BANK_GSM" \
      --bank_dump "$BANK_GSM"
  else
    echo "[GSM8K] Phase 2/2 skipped (TOTAL <= SEED)"
  fi
}

case "$TASK" in
  csqa)  run_csqa ;;
  gsm8k) run_gsm8k ;;
  both)  run_csqa; run_gsm8k ;;
  *)     echo "Unknown TASK=$TASK (use: csqa|gsm8k|both)"; exit 1 ;;
esac

echo "Step0 done."
