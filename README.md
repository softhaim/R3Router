Here’s a concise, paper-submission–style README in English.

---

# R³Route: Subquery-Level Reflective Routing

R³Route is a **subquery-level reflective routing** framework. A small language model (SLM) first drafts sub-answers; then a GRU-based router decides, per subquery, whether to **Retain** the draft, **Revise** it by delegating to a selected LLM, or **Rewrite** the whole answer by routing the original query to an LLM. This avoids token-level switching overhead while keeping query-level fallback when needed.

## Repository Layout

* `step0.py` — **Data generation**: builds routing episodes (JSONL) with acceptable action/model sets.
* `step1.py` — **Training**: SFT → RL/GRPO for the router (BERT encoder frozen by default).
* `step2.py` — **Inference & evaluation**: runs the trained router and reports metrics.
* `run_step0.sh`, `run_train.sh`, `run_step2.sh` — Minimal shell wrappers for each stage.

## Quick Start

```bash
# 0) Generate data
bash run_step0.sh

# 1) Train (SFT → RL/GRPO)
bash run_train.sh

# 2) Inference & evaluation
bash run_step2.sh
```

## Requirements

* Python 3.10+
* PyTorch, Transformers, NumPy (install via pip)

## Notes

* The **SLM must be model index 0**. It is only valid for the **Retain/Continue** action (not for Revise/Rewrite).
* Checkpoint files may include the frozen encoder weights, so sizes can be large.
* Paths, model lists, and hyperparameters are specified in the provided shell scripts; adjust to your environment.

---
