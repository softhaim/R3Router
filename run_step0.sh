set -euo pipefail
# 맨 위쪽 공통 설정 근처
export PYTHONUNBUFFERED=1

# ===== 사용자 설정(환경변수로 덮어쓰기 가능) =====
PY=${PY:-step0_CSQA.py}
TASK=${TASK:-csqa}        # both | csqa | gsm8k

# CSQA
CSQA_TOTAL="${CSQA_TOTAL:-4000}"
CSQA_SEED="${CSQA_SEED:-500}"
OUT_CSQA="${OUT_CSQA:-runs/csqa_train.jsonl}"
BANK_CSQA="${BANK_CSQA:-banks/bank_csqa.jsonl}"

# GSM8K
GSM_NUM="${GSM_NUM:-4000}"
OUT_GSM="${OUT_GSM:-runs/gsm8k_train.jsonl}"
RET_BANK_GSM_READ="${RET_BANK_GSM_READ:-bank_dump.jsonl}"   # 참조(없어도 안전)
RET_BANK_GSM_WRITE="${RET_BANK_GSM_WRITE:-bank_ret.jsonl}"   # 덤프

# 공통(GPU/버퍼링)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1  # tqdm 갱신 안정화

# 디렉토리 준비(읽기용 파일의 dirname은 . 일 수 있지만 mkdir -p . 는 무해)
mkdir -p \
  "$(dirname "$OUT_CSQA")" \
  "$(dirname "$OUT_GSM")" \
  "$(dirname "$BANK_CSQA")" \
  "$(dirname "$RET_BANK_GSM_WRITE")"

echo "==== Config ===="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TASK=$TASK"
echo "[CSQA] total=$CSQA_TOTAL, seed=$CSQA_SEED, out=$OUT_CSQA, bank=$BANK_CSQA"
echo "[GSM8K] num=$GSM_NUM, out=$OUT_GSM, ret_read=$RET_BANK_GSM_READ, ret_write=$RET_BANK_GSM_WRITE"
echo "==============="

run_csqa_seed () {
  echo "[CSQA] Phase 1: seed bank build ($CSQA_SEED)"
  python "$PY" \
    --task csqa \
    --csqa_offset 0 \
    --num_samples "$CSQA_SEED" \
    --out "$OUT_CSQA" \
    --bank_dump "$BANK_CSQA"
}

run_csqa_main () {
  local REMAIN=$(( CSQA_TOTAL - CSQA_SEED ))
  if (( REMAIN > 0 )); then
    echo "[CSQA] Phase 2: train with retrieval (offset=$CSQA_SEED, num=$REMAIN)"
    python "$PY" \
      --task csqa \
      --csqa_offset "$CSQA_SEED" \
      --num_samples "$REMAIN" \
      --out "$OUT_CSQA" \
      --ret_bank "$BANK_CSQA" \
      --bank_dump "$BANK_CSQA"
  else
    echo "[CSQA] Phase 2 skipped (REMAIN <= 0)"
  fi
}

run_gsm8k () {
  echo "[GSM8K] Train using ret_bank=$RET_BANK_GSM_READ, dump to $RET_BANK_GSM_WRITE"
  python "$PY" \
    --task gsm8k \
    --num_samples "$GSM_NUM" \
    --out "$OUT_GSM" \
    --ret_bank "$RET_BANK_GSM_READ" \
    --bank_dump "$RET_BANK_GSM_WRITE"
}

case "$TASK" in
  csqa)
    run_csqa_seed
    run_csqa_main
    ;;
  gsm8k)
    run_gsm8k
    ;;
  both)
    run_csqa_seed
    run_csqa_main
    run_gsm8k
    ;;
  *)
    echo "Unknown TASK=$TASK (use: both|csqa|gsm8k)"; exit 1;;
esac

echo "All done."
