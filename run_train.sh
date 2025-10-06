#!/usr/bin/env bash
set -e

python step1.py \
  --data train_math_500.jsonl \
  --save_name router_rl_MATH_GRPO.pt \
  --models SLM Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-14B-Instruct \
  --rl_alg grpo --group_size 4 --baseline loo --reward_norm group \
  --kl_coef 0.25 --ce_coef 0.10 --ent_coef 0.01
