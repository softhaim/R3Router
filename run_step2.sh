#!/usr/bin/env bash
set -euo pipefail
mkdir -p log

ts=$(date +%F_%H%M%S)

cmd_gsm8k_GRPO=(python -u step2.py --task gsm8k --split test --limit 1319 \
  --out gsm8k_result.jsonl\
  --router_ckpt ckpts/router_rl_GSM8k_GRPO.pt \
  --fp32_models "Qwen/Qwen2.5-7B-Instruct" \
  --llm_only_mode chain \
  --charge_slm_to_router \
  --router_only 

cmd_csqa_GRPO=(python -u step2.py --task csqa --split validation --limit 1221 \
  --out CSQA_result.jsonl\
  --router_ckpt ckpts/router_rl_CSQA_GRPO.pt \
  --fp32_models "Qwen/Qwen2.5-7B-Instruct" \
  --llm_only_mode chain \
  --charge_slm_to_router \
  --router_only 

cmd_csqa_GRPO_200=(python -u step2.py --task csqa --split validation --limit 200 \
  --out CSQA_200sample.jsonl\
  --router_ckpt ckpts/router_rl_CSQA_GRPO.pt \
  --fp32_models "Qwen/Qwen2.5-7B-Instruct" \
  --llm_only_mode chain \
  --charge_slm_to_router 



run() {
  local name="$1"; shift
  local logfile="log/${name}_${ts}.log"
  echo "[$(date +'%F %T')] START ${name} -> ${logfile}"
  "$@" >"$logfile" 2>&1
  local rc=$?
  echo "[$(date +'%F %T')] DONE  ${name} (rc=${rc})"
  return $rc
}

# export CUDA_VISIBLE_DEVICES=0

run csqa_GRPO  "${cmd_csqa_GRPO[@]}"
sleep 10
run gsm8k_GRPO "${cmd_gsm8k_GRPO[@]}"
