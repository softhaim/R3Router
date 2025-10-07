#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified step-level data generator for GSM8K (math) and CSQA (MCQA).

Key updates:
- Persistent model cache (ModelPool) to avoid re-loading models every sample.
- Stable tqdm progress bar with explicit pbar.update(1) and quiet logging via tqdm.write().
- Suppress noisy HF logs & tokenizer warnings.
- Fixed csqa_list scope bug (`total_n = len(csqa_list)` inside CSQA branch).
"""

import os, sys, argparse, json, re, math, hashlib, time, gc, uuid, random, traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# --- Quiet noisy libs ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES","2,3"))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from datasets import load_dataset
from tqdm import tqdm
from contextlib import contextmanager

# ===== 비용표(간소) =====
MODEL_COST_CONFIG = {
    'qwen25_1p5b_instruct':       {'input_cost': 0.1e-6,   'output_cost': 0.2e-6},
    'qwen25_7b_instruct':         {'input_cost': 0.1e-6,   'output_cost': 0.2e-6},
    'qwen25_14b_instruct':        {'input_cost': 0.15e-6,  'output_cost': 0.30e-6},
    'llama31_8b_instruct':        {'input_cost': 0.1e-6,   'output_cost': 0.2e-6},
}

FREE_MODELS: set[str] = set()

def is_free_model(model_name: str) -> bool:
    return model_name.lower() in FREE_MODELS

def cost_key_from_model_name(n:str)->Optional[str]:
    n=n.lower()
    if "qwen" in n and ("2.5" in n or "2-5" in n or "2_5" in n):
        if "1.5b" in n or "1_5b" in n: return "qwen25_1p5b_instruct"
        if "7b" in n: return "qwen25_7b_instruct"
        if "14b" in n: return "qwen25_14b_instruct"
    if "llama-3.1" in n or "llama3.1" in n:
        if "8b" in n: return "llama31_8b_instruct"
    return None

def apply_api_cost(cost: Dict[str,float], model_name: str)->None:
    if is_free_model(model_name):
        cost['api_cost'] = 0.0
        return
    key = cost_key_from_model_name(model_name)
    if key and key in MODEL_COST_CONFIG:
        ic = MODEL_COST_CONFIG[key]['input_cost']; oc = MODEL_COST_CONFIG[key]['output_cost']
        pt = int(cost.get("prompt_tokens",0)); ct = int(cost.get("completion_tokens",0))
        cost['api_cost'] = float(ic*pt + oc*ct)
    else:
        cost['api_cost'] = 0.0

# ===== 시드/유틸 =====
def set_global_seed(seed:int=0)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

NUM_WITH_COMMAS = re.compile(r'-?\d[\d,]*(?:\.\d+)?')

def last_number(text:str)->Optional[str]:
    if not text: return None
    toks = NUM_WITH_COMMAS.findall(str(text))
    if not toks: return None
    for tok in reversed(toks):
        t = tok.replace(",", "")
        if len(t) >= 1: return t
    return toks[-1].replace(",","")

def now_ts()->float:
    return float(time.time())

def _norm(s: str) -> str:
    s = re.sub(r'[`*_]+', '', s or '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _is_qwen25_7b(name: str) -> bool:
    n = (name or "").lower().replace(" ", "")
    return ("qwen" in n) and (("2.5" in n) or ("2-5" in n) or ("2_5" in n)) and ("7b" in n)

def resolve_dtype_for_model(model_name: str, slm_model: str, slm_dtype: str, llm_dtype: str) -> str:
    # SLM은 SLM dtype 고정
    if (model_name or "").lower() == (slm_model or "").lower():
        return slm_dtype
    # Qwen2.5 7B만 FP32 강제
    if _is_qwen25_7b(model_name):
        return "float32"
    # 그 외는 기본 LLM dtype (예: float16)
    return llm_dtype

# ===== HF Chat =====
def _get_hf_token():
    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or None)

class HFChatCaller:
    def __init__(self, model_name: str, dtype: Optional[str]="bfloat16", hf_token: Optional[str]=None, seed:Optional[int]=None, device_map:str="auto"):
        self.model_name = model_name
        if str(dtype).lower() in ("float32","fp32"): dt = torch.float32
        elif str(dtype).lower() in ("bfloat16","bf16"): dt = torch.bfloat16
        else: dt = torch.float16
        tok = hf_token if hf_token is not None else _get_hf_token()
        auth = ({'token': tok, 'use_auth_token': tok} if tok else {})
        last_err=None; self.tokenizer=None
        for use_fast in (True, False):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=use_fast, **auth)
                break
            except Exception as e: last_err=e
        if self.tokenizer is None: raise last_err
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **auth)
        fm = AutoModelForSeq2SeqLM if getattr(cfg, "is_encoder_decoder", False) else AutoModelForCausalLM
        self.model = fm.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dt, device_map=device_map, low_cpu_mem_usage=True, **auth
        )
        self.model.eval()
        self.seed = seed
        self.is_encdec = getattr(cfg, "is_encoder_decoder", False)

    def _apply_template(self, messages: List[Dict[str,str]])->str:
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages])+"\nASSISTANT:"

    def chat(self, messages: List[Dict[str,str]], max_new_tokens:int=512, temperature:float=0.0, top_p:float=0.95, seed:Optional[int]=None)->Tuple[str,Dict[str,float]]:
        prompt = self._apply_template(messages)
        if seed is None: seed = self.seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k:v.to(dev) for k,v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs = dict(max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=False,
                          pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        if temperature > 0:
            gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=float(top_p)))
        else:
            gen_kwargs.update(dict(do_sample=False))
        t0=time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter()-t0)*1000.0
        seq = out.sequences[0]
        gen_part = seq if self.is_encdec else seq[prompt_len:]
        text = self.tokenizer.decode(gen_part, skip_special_tokens=True).strip()
        cost = {"prompt_tokens": int(prompt_len), "completion_tokens": int(gen_part.numel()), "latency_ms": float(latency_ms)}
        apply_api_cost(cost, self.model_name)
        return text, cost

    def close(self):
        try: del self.model; del self.tokenizer
        except: pass
        gc.collect(); torch.cuda.empty_cache()

# ===== 모델 풀 (캐시) =====
class ModelPool:
    def __init__(self, device_map:str="auto"):
        self._pool: Dict[Tuple[str,str], HFChatCaller] = {}
        self._device_map = device_map
    def get(self, name:str, *, dtype:str="float16", seed:int=0)->HFChatCaller:
        key=(name, dtype.lower())
        if key in self._pool: return self._pool[key]
        caller = HFChatCaller(name, dtype=dtype, seed=seed, device_map=self._device_map)
        tqdm.write(f"[ModelPool] loaded: {name} (dtype={dtype})")
        self._pool[key]=caller
        return caller
    def close_all(self):
        for c in list(self._pool.values()):
            try: c.close()
            except: pass
        self._pool.clear()
        gc.collect(); torch.cuda.empty_cache()

# ===== 공통 파서 =====
_ANS_PATTERNS = [r'^\s*(?:\d+[\.\)]\s*)?ANS(?:WER)?\s*[:=]\s*(.+?)\s*$']
_FINAL_PATTERNS = [
    r'^\s*FINAL(?:\s*ANSWER)?\s*[:=]\s*([A-Ea-e0-9\.\-]+)\s*$',
    r'^\s*Final\s*[:=]\s*([A-Ea-e0-9\.\-]+)\s*$',
]

def extract_all_ans_lines(text: str) -> list[str]:
    vals = []
    for raw in (text or "").splitlines():
        line = re.sub(r'\s+', ' ', raw or '').strip()
        if 'ANS' not in line.upper(): continue
        m = re.search(r'(ANS(?:WER)?\s*[:=]\s*)(.+)$', line, flags=re.I)
        if not m: continue
        v = re.sub(r'(?i)^ANS(?:WER)?\s*[:=]\s*', '', m.group(0)).strip()
        v = re.sub(r'\s*(FINAL(?:\s*ANSWER)?\s*[:=].*)$', '', v, flags=re.I).strip()
        if len(v)>=2 and ((v[0],v[-1]) in {('"','"'),("'", "'"),("`","`")}): v=v[1:-1].strip()
        if v: vals.append(v)
    return vals

def extract_first_ans_line(text: str) -> str:
    vals = extract_all_ans_lines(text)
    return vals[0] if vals else ""

def extract_last_ans_line(text: str) -> str:
    vals = extract_all_ans_lines(text)
    return vals[-1] if vals else ""

def extract_final_raw(text: str) -> Optional[str]:
    matches=[]
    for p in _FINAL_PATTERNS:
        matches += list(re.finditer(p, text or "", flags=re.I|re.M))
    if not matches: return None
    return matches[-1].group(1).strip()

def extract_final_number(text: str) -> str:
    f = extract_final_raw(text)
    if f and re.fullmatch(r'-?\d+(?:\.\d+)?', f): return f
    nums = re.findall(r'-?\d+(?:\.\d+)?', text or "")
    return nums[-1] if nums else ""

def extract_final_letter(text: str, labels: List[str]=list("ABCDE")) -> str:
    f = extract_final_raw(text) or ""
    m = re.search(r'\b([A-E])\b', f.upper())
    if m and m.group(1) in labels: return m.group(1)
    m2 = re.search(r'\b([A-E])\b', (text or "").upper())
    return m2.group(1) if (m2 and m2.group(1) in labels) else ""

def judge_numeric(gold:str, pred:str)->bool:
    def _pick(s):
        m = NUM_WITH_COMMAS.findall(str(s))
        return m[-1].replace(",","") if m else None
    g = _pick(gold); p = _pick(pred)
    if g is not None and p is not None: return g == p
    return str(gold).strip() == str(pred).strip()

def judge_letter(gold_letter: str, pred_letter: str) -> bool:
    return (gold_letter or "").strip().upper() == (pred_letter or "").strip().upper()

def make_bank_item(source_id: str, step_idx: int, subq: str, resp: str, ans: str,
                   action: Optional[str], model: Optional[str]) -> Dict[str, Any]:
    key_src = f"{_norm(subq)}\n{_norm(resp)}"
    return {
        "id": str(uuid.uuid4()),
        "subq": subq,
        "resp": resp,
        "ans": ans,
        "source": source_id,
        "step_idx": step_idx,
        "ts": now_ts(),
        "hash_key": hashlib.md5(key_src.encode("utf-8")).hexdigest(),
        "action": action,
        "model": model,
    }

# ===== 프롬프트 =====
def prompt_decompose_math(problem:str, qtype:str)->List[Dict[str,str]]:
    user = f"""I will give you a math problem (type: {qtype}). Break it into EXACTLY 3 minimal sub-problems (≤10 words each).
- Favor counts over percentages; include all given numbers at least once.
- Order them by how to solve.

Question:
{problem}

Answer Format (STRICT):
To solve "xxx", we need to:
1. ...
2. ...
3. ..."""
    return [{"role":"system","content":"Break the problem into exactly 3 atomic numeric steps."},
            {"role":"user","content":user}]

def prompt_decompose_csqa(question:str, options_str:str)->List[Dict[str,str]]:
    user = f"""I have a single-choice commonsense question. Decompose the reasoning into EXACTLY 3 concise sub-problems (≤14 words), ordered by solving.
Question:
{question}

Choices:
{options_str}

Format (STRICT):
To solve "xxx", we need to clarify / solve:
1. ...
2. ...
3. ..."""
    return [{"role":"system","content":"You break commonsense questions into 3 actionable sub-problems."},
            {"role":"user","content":user}]

def parse_decompose(text:str, fallback_steps:List[str])->List[str]:
    m = re.search(r'(?:clarify\s*/\s*solve|do|we need to)\s*:\s*', text, flags=re.I)
    sub = text[m.end():] if m else text
    outs=[]
    for raw in (sub or "").splitlines():
        ln = re.sub(r'^\s*(\d+[\.\)]\s*|step\s*\d+\s*[:.)-]\s*|[-*•]\s*)','', raw.strip(), flags=re.I)
        if 2 <= len(ln.split()) <= 16:
            outs.append(ln)
        if len(outs)>=3: break
    return (outs + fallback_steps)[:3]

def prompt_edit_step_minimal_with_support(
    subtask: str,
    current_step_text: str,
    support_pairs: List[Tuple[str, str]] = None,
    *,
    problem_text: Optional[str] = None,
    solved_prefix_texts: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    support_pairs = (support_pairs or [])[:2]
    ctx = ""
    if problem_text:
        ctx += f"Original problem:\n{problem_text}\n\n"
    if solved_prefix_texts:
        ctx += "Fixed earlier steps (DO NOT CHANGE):\n"
        for i, txt in enumerate(solved_prefix_texts, start=1):
            ctx += f"- Step {i}: {txt}\n"
        ctx += "\n"
    support_block = ""
    if support_pairs:
        support_block = "Style examples (DO NOT copy content; format only):\n<EXAMPLES>\n"
        for i,(sq,resp) in enumerate(support_pairs,1):
            snippet = (resp or "").strip()
            snippet = re.sub(r'(?im)^FINAL\s*:.*$', '', snippet)
            if len(snippet) > 400: snippet = snippet[:400].rstrip() + " ..."
            support_block += f"[{i}] Subproblem: {sq}\nResponse:\n{snippet}\n\n"
        support_block += "</EXAMPLES>\n"
    user = f"""EDIT with minimal changes and return EXACTLY TWO lines:
Line 1: one short sentence solving THIS subproblem only.
Line 2: ANS: <short answer>

{subtask=}
{ctx}Current step response (two-line form):
{current_step_text}

{support_block}Hard Rules:
- Output EXACTLY TWO lines. No headers or blanks.
- Keep problem constants unchanged; do not invent facts.
- Never output 'FINAL'."""
    return [{"role":"user","content":user}]

# --- NEW: one-step repair prompt (exactly two lines, no FINAL) ---
def prompt_repair_single_step(
    subtask: str,
    draft_text: str,
    *,
    problem_text: Optional[str] = None,
    solved_prefix_texts: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    ctx = ""
    if problem_text:
        ctx += f"Original problem:\n{problem_text}\n\n"
    if solved_prefix_texts:
        ctx += "Fixed earlier steps (DO NOT CHANGE):\n"
        for i, txt in enumerate(solved_prefix_texts, start=1):
            ctx += f"- Step {i}: {txt}\n"
        ctx += "\n"

    user = f"""Rewrite the draft for THIS subproblem into STRICT two-line format.

{subtask=}
{ctx}Draft:
\"\"\"{(draft_text or '').strip()[:800]}\"\"\"

Return EXACTLY TWO lines:
Line 1: one short sentence solving THIS subproblem only.
Line 2: ANS: <short answer>

Hard Rules:
- Output EXACTLY TWO lines. No headers or blanks.
- Keep problem constants unchanged; do not invent facts.
- Never output 'FINAL'."""
    return [{"role":"user","content":user}]


# --- NEW: format enforcement (two lines + numeric ANS for math) ---
_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')

def _take_two_lines(text: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) >= 2:
        # pick first line + the last line that contains 'ANS'
        ans_line_idx = None
        for idx in range(len(lines)-1, -1, -1):
            if 'ANS' in lines[idx].upper():
                ans_line_idx = idx
                break
        if ans_line_idx is not None:
            return lines[0], lines[ans_line_idx]
        return lines[0], lines[1]
    if len(lines) == 1:
        return lines[0], ""
    return "", ""

def _is_numeric_strict(s: str) -> bool:
    return bool(re.fullmatch(r'-?\d+(?:\.\d+)?', (s or "").strip()))

def _extract_last_number(text: str) -> str:
    nums = _NUM_RE.findall(text or "")
    return nums[-1] if nums else ""

def enforce_two_line_and_numeric_for_math(
    text: str,
    *,
    fallback_hint_text: Optional[str] = None
) -> Tuple[str, str, bool]:
    """
    Returns: (fixed_text, ans_value, changed)
    - Ensures EXACTLY two lines and "ANS: <number>" for math.
    - If ANS is missing/non-numeric, try last number fallback (from text or hint).
    """
    l1, l2 = _take_two_lines(text)
    changed = False

    # ensure 'ANS:' line
    ans_raw = ""
    m = re.search(r'(?i)\bANS(?:WER)?\s*[:=]\s*(.+)$', l2)
    if m: ans_raw = m.group(1).strip()

    # numeric check
    if not _is_numeric_strict(ans_raw):
        cand = _extract_last_number(l2) or _extract_last_number(text) or _extract_last_number(fallback_hint_text or "")
        if cand:
            l2 = f"ANS: {cand}"
            ans_raw = cand
            changed = True

    # if still not two lines or missing ANS, try to create minimal
    if not l1:
        # salvage a short rationale from original text
        l1 = "(short rationale)"
        changed = True
    if 'ANS:' not in l2.upper():
        cand = _extract_last_number(text) or _extract_last_number(fallback_hint_text or "")
        l2 = f"ANS: {cand}" if cand else "ANS: "
        ans_raw = cand or ""
        changed = True

    fixed = f"{l1}\n{l2}"
    return fixed, ans_raw, changed


def build_chain_prompt_strict_math(problem:str, steps:List[str],
                                   solved_prefix:List[Tuple[int,str]],
                                   start_idx:int, include_step_t:bool, step_t_text:Optional[str]) -> List[Dict[str,str]]:
    prefix_block=""
    if solved_prefix:
        prefix_block+="Solved subproblems so far (DO NOT CHANGE):\n"
        for i,txt in solved_prefix:
            prefix_block+=f"- Step {i+1}: (fixed)\n{txt}\n"
    tline=""
    if include_step_t and step_t_text:
        tline=f"- Step {start_idx+1}: (fixed by policy)\n{step_t_text}\n"
    guide = """FORMAT (STRICT):
- For each remaining subproblem EXCEPT the last one: output EXACTLY TWO lines:
  Line 1: one short rationale sentence for THIS subproblem.
  Line 2: ANS: <short answer>
- For the LAST subproblem: output EXACTLY THREE lines:
  Line 1: one short rationale sentence.
  Line 2: ANS: <short answer>
  Line 3: FINAL: <number>"""
    user=f"""You are given a math problem and its subproblems.

Problem:
{problem}

Subproblems (ordered):
{chr(10).join([f"{i+1}. {s}" for i,s in enumerate(steps)])}

{prefix_block}{tline}
Now write the answers for the remaining subproblems in order.

{guide}
"""
    return [{"role":"system","content":"You solve math step-by-step and obey the format strictly."},
            {"role":"user","content":user}]

def build_chain_prompt_strict_csqa(problem_with_options:str, steps:List[str],
                                   solved_prefix:List[Tuple[int,str]],
                                   start_idx:int, include_step_t:bool, step_t_text:Optional[str]) -> List[Dict[str,str]]:
    prefix_block=""
    if solved_prefix:
        prefix_block+="Solved subproblems so far (DO NOT CHANGE):\n"
        for i,txt in solved_prefix:
            prefix_block+=f"- Step {i+1}: (fixed)\n{txt}\n"
    tline=""
    if include_step_t and step_t_text:
        tline=f"- Step {start_idx+1}: (fixed by policy)\n{step_t_text}\n"
    guide = """FORMAT (STRICT):
- For each remaining subproblem EXCEPT the last one: output EXACTLY TWO lines:
  Line 1: one short rationale for THIS subproblem.
  Line 2: ANS: <short answer>
- For the LAST subproblem: output EXACTLY THREE lines:
  Line 1: one short rationale.
  Line 2: ANS: <short answer>
  Line 3: FINAL: <LETTER>   # A/B/C/D/E only"""
    user=f"""You are given a commonsense single-choice question (MCQA), its choices, and decomposed subproblems.

{problem_with_options}

Subproblems (ordered):
{chr(10).join([f"{i+1}. {s}" for i,s in enumerate(steps)])}

{prefix_block}{tline}
Now write the answers for the remaining subproblems in order.

{guide}
"""
    return [{"role":"system","content":"You reason step-by-step, obeying a strict format."},
            {"role":"user","content":user}]

def prompt_llm_only_math(problem:str)->List[Dict[str,str]]:
    user=f"""Solve the following math problem carefully. Show brief reasoning and END WITH EXACTLY ONE LINE:
FINAL: <number>  (digits only)

Problem:
{problem}"""
    return [{"role":"system","content":"You are a careful math solver."},{"role":"user","content":user}]

def prompt_llm_only_csqa(problem_with_options:str)->List[Dict[str,str]]:
    user=f"""Solve the single-choice question carefully and END WITH EXACTLY ONE LINE:
FINAL: <LETTER>   # A/B/C/D/E only

{problem_with_options}"""
    return [{"role":"system","content":"You answer MCQA with a single-letter final line."},
            {"role":"user","content":user}]

def prompt_finalize_attach_direct_math(problem: str, steps: List[str], executed_chunks: List[str]) -> List[Dict[str, str]]:
    lines=[]
    for i,ch in enumerate(executed_chunks,1):
        m=re.search(r'\bANS(?:WER)?\s*[:=]\s*([^\n]+)', (ch or ""), flags=re.I)
        ans=(m.group(1).strip() if m else "UNSURE")
        lines.append(f"- Step {i}: {steps[i-1]} -> ANS: {ans}")
    u=("Solve the problem carefully and provide ONLY the final numeric answer.\n\n"
       f"Problem:\n{problem}\n\nHelpful notes:\n"+"\n".join(lines)+
       "\n\nEND WITH EXACTLY ONE LINE:\nFINAL: <number>")
    return [{"role":"system","content":"You are a careful math solver."},
            {"role":"user","content":u}]

def prompt_finalize_attach_direct_csqa(problem_with_options: str, steps: List[str], executed_chunks: List[str]) -> List[Dict[str, str]]:
    lines=[]
    for i,ch in enumerate(executed_chunks,1):
        m=re.search(r'\bANS(?:WER)?\s*[:=]\s*([^\n]+)', (ch or ""), flags=re.I)
        ans=(m.group(1).strip() if m else "UNSURE")
        lines.append(f"- Step {i}: {steps[i-1]} -> ANS: {ans}")
    u=("Synthesize ONLY from the given sub-results (no recomputation) and output the letter.\n\n"
       f"{problem_with_options}\n\nNotes:\n"+"\n".join(lines)+
       "\n\nEND WITH EXACTLY ONE LINE:\nFINAL: <LETTER>")
    return [{"role":"system","content":"You output only a single letter."},
            {"role":"user","content":u}]

def prompt_force_final_only_math(problem:str, prev_text:str)->List[Dict[str,str]]:
    user=f"""You wrote the following reasoning for the problem below.

Problem:
{problem}

Your text:
\"\"\"{(prev_text or '').strip()[:1200]}\"\"\"

Now output EXACTLY ONE LINE, NOTHING ELSE:
FINAL: <number>"""
    return [{"role":"user","content":user}]

def prompt_force_final_only_csqa(problem_with_options:str, prev_text:str)->List[Dict[str,str]]:
    user=f"""You wrote the following reasoning for the question below.

{problem_with_options}

Your text:
\"\"\"{(prev_text or '').strip()[:1200]}\"\"\"

Now output EXACTLY ONE LINE, NOTHING ELSE:
FINAL: <LETTER>"""
    return [{"role":"user","content":user}]

# ===== 데이터셋 I/O =====
def format_mcqa_problem(q:str, labels:List[str], texts:List[str])->str:
    parts=[f"Question:\n{q}\n\nChoices:"]
    for lab,tx in zip(labels,texts):
        parts.append(f"{lab}. {tx}")
    return "\n".join(parts)

# ===== 두 줄 리포맷(공통) =====
def prompt_reformat_two_lines(problem_block:str, steps:List[str], prev_text:str, *, last_is_letter:bool)->List[Dict[str,str]]:
    user = f"""Rewrite the previous draft into strict per-subproblem outputs.

{problem_block}

Subproblems:
1) {steps[0]}
2) {steps[1]}
3) {steps[2]}

Rules:
- For each subproblem: EXACTLY TWO lines
  Line 1: one short rationale sentence
  Line 2: ANS: <short answer>
- Do not add 'FINAL' here.
- Keep constants unchanged.

Previous draft:
\"\"\"{prev_text[:3000]}\"\"\""""
    return [{"role":"system","content":"You rewrite into strict two-line answers per subproblem."},
            {"role":"user","content":user}]

def reformat_two_line_steps(slm:HFChatCaller, problem_block:str, steps:List[str], prev_text:str, *, last_is_letter:bool)->Tuple[List[str], Dict[str,float], str]:
    msgs = prompt_reformat_two_lines(problem_block, steps, prev_text, last_is_letter=last_is_letter)
    text, cost = slm.chat(msgs, max_new_tokens=400, temperature=0.0)
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    out=[]
    for b in blocks:
        lines=[ln for ln in b.splitlines() if ln.strip()]
        if len(lines)>=2:
            out.append(f"{lines[0].strip()}\n{lines[1].strip()}")
    if len(out)<len(steps):
        ans = extract_all_ans_lines(text)
        while len(out)<len(steps):
            a = ans[len(out)] if len(ans)>len(out) else ""
            out.append(f"(short rationale)\nANS: {a}")
    return out[:len(steps)], cost, text

# ===== 체인 러너 =====
def run_singlepass_chain_math(slm, problem, steps):
    msgs = build_chain_prompt_strict_math(problem, steps, solved_prefix=[],
                              start_idx=0, include_step_t=False, step_t_text=None)
    text, cost = slm.chat(msgs, max_new_tokens=900, temperature=0.0)
    final = extract_final_number(text)
    all_ans = extract_all_ans_lines(text)
    N, K = len(steps), len(all_ans)
    ans_list = ([""]*(N-K) + all_ans) if K < N else all_ans[-N:]
    return ans_list, str(final), cost, text

def run_singlepass_chain_csqa(slm, problem_block, steps):
    msgs = build_chain_prompt_strict_csqa(problem_block, steps, solved_prefix=[],
                              start_idx=0, include_step_t=False, step_t_text=None)
    text, cost = slm.chat(msgs, max_new_tokens=900, temperature=0.0)
    final = extract_final_letter(text)
    all_ans = extract_all_ans_lines(text)
    N, K = len(steps), len(all_ans)
    ans_list = ([""]*(N-K) + all_ans) if K < N else all_ans[-N:]
    return ans_list, str(final), cost, text

def build_chain_msgs(task:str, problem_or_block:str, steps:List[str],
                     solved_prefix:List[Tuple[int,str]], start_idx:int,
                     include_step_t:bool, step_t_text:str):
    if task=="csqa":
        return build_chain_prompt_strict_csqa(problem_or_block, steps, solved_prefix, start_idx, include_step_t, step_t_text)
    else:
        return build_chain_prompt_strict_math(problem_or_block, steps, solved_prefix, start_idx, include_step_t, step_t_text)

# --- REPLACE: run_finalize_auto (auto-select attach_direct vs direct) ---
def _chunk_has_numeric_ans(ch: str) -> bool:
    if not ch: return False
    m = re.search(r'(?i)\bANS(?:WER)?\s*[:=]\s*([^\n]+)', ch or "")
    if not m: return False
    val = (m.group(1) or "").strip()
    return bool(re.fullmatch(r'-?\d+(?:\.\d+)?', val))

def _steps_look_reliable(executed_chunks: list, *, task: str) -> bool:
    if not executed_chunks or len(executed_chunks) < 3: return False
    ok = []
    for ch in executed_chunks[:3]:
        if not ch or len(ch) > 6000: ok.append(False); continue
        if task == "csqa":
            # non-empty ANS is good enough for sub-results
            m = re.search(r'(?i)\bANS(?:WER)?\s*[:=]\s*([^\n]+)', ch or "")
            ok.append(bool(m and (m.group(1) or "").strip()))
        else:
            ok.append(_chunk_has_numeric_ans(ch))
    return all(ok)

def run_finalize_auto(task:str, problem_or_block:str, steps: list, executed_chunks: list,
                      caller: HFChatCaller, *, max_new_tokens: int = 80):
    # choose mode by sub-step reliability
    reliable = _steps_look_reliable(executed_chunks, task=task)
    if task == "csqa":
        if reliable:
            msgs = prompt_finalize_attach_direct_csqa(problem_or_block, steps, executed_chunks)
            text, cost = caller.chat(msgs, max_new_tokens=max_new_tokens, temperature=0.0)
        else:
            msgs = prompt_llm_only_csqa(problem_or_block)
            text, cost = caller.chat(msgs, max_new_tokens=180, temperature=0.0)
        final = extract_final_letter(text)
        if not final:
            text2, cost2 = caller.chat(prompt_force_final_only_csqa(problem_or_block, text), max_new_tokens=16, temperature=0.0)
            f2 = extract_final_letter(text2)
            if f2: final = f2
            for k in cost: cost[k] += cost2.get(k, 0.0)
        return (final or ""), cost
    else:
        if reliable:
            msgs = prompt_finalize_attach_direct_math(problem_or_block, steps, executed_chunks)
            text, cost = caller.chat(msgs, max_new_tokens=max_new_tokens, temperature=0.0)
        else:
            msgs = prompt_llm_only_math(problem_or_block)
            text, cost = caller.chat(msgs, max_new_tokens=180, temperature=0.0)
        final = extract_final_number(text)
        if not final:
            text2, cost2 = caller.chat(prompt_force_final_only_math(problem_or_block, text), max_new_tokens=16, temperature=0.0)
            f2 = extract_final_number(text2)
            if f2: final = f2
            for k in cost: cost[k] += cost2.get(k, 0.0)
        return (final or ""), cost
    
def run_detract_then_slm_with_finalize(
    task:str,
    slm:HFChatCaller, detract:HFChatCaller, finalizer:HFChatCaller,
    problem_block:str, steps:List[str], base_fixed_texts:List[str], t:int,
    support_pairs:Optional[List[Tuple[str,str]]]=None
)->Tuple[str, Dict[str,float], Dict[str,float], Dict[str,float], str, str, str, List[str]]:
    # 1) First try: minimal edit (two lines)
    d_msgs = prompt_edit_step_minimal_with_support(
        steps[t], base_fixed_texts[t], support_pairs or [],
        problem_text=problem_block,
        solved_prefix_texts=[base_fixed_texts[i] for i in range(t)]
    )
    d_text, d_cost = detract.chat(d_msgs, max_new_tokens=200, temperature=0.0)

    # 2) If format not clean, do one repair pass (cheap, deterministic)
    l1, l2 = _take_two_lines(d_text)
    need_repair = (not l1) or ('ANS' not in l2.upper())
    if need_repair:
        r_msgs = prompt_repair_single_step(
            steps[t], d_text,
            problem_text=problem_block,
            solved_prefix_texts=[base_fixed_texts[i] for i in range(t)]
        )
        r_text, r_cost = detract.chat(r_msgs, max_new_tokens=120, temperature=0.0)
        # accumulate cost
        for k in d_cost: d_cost[k] = d_cost.get(k, 0.0) + r_cost.get(k, 0.0)
        d_text = r_text

    if task == "csqa":
        l1, l2 = _take_two_lines(d_text)
        if l1 or l2:
            d_text = f"{l1}\n{l2}"

    # 3) Math: force numeric ANS if missing / non-numeric (match inference fallback)
    if task != "csqa":
        fixed, ans_raw, changed = enforce_two_line_and_numeric_for_math(
            d_text,
            fallback_hint_text=base_fixed_texts[t]
        )
        if changed:
            d_text = fixed
    d_ans = extract_last_ans_line(d_text)

    # 4) Roll out the remainder of the chain from step t
    tail_runner = detract if (t == len(steps) - 1) else slm
    solved_prefix = [(i, base_fixed_texts[i]) for i in range(t)] + [(t, d_text)]
    msgs = build_chain_msgs(task, problem_block, steps,
                            solved_prefix=solved_prefix,
                            start_idx=t, include_step_t=True, step_t_text=d_text)
    tail_text, tail_cost = tail_runner.chat(msgs, max_new_tokens=650, temperature=0.0)
    final_tail = extract_final_letter(tail_text) if task=="csqa" else extract_final_number(tail_text)

    # 5) Compose executed chunks after step t (prefix fixed, t replaced by detract, rest from tail)
    executed_chunks_after=[]
    later_pairs=[]
    if t+1 <= len(steps)-1:
        vals = extract_all_ans_lines(tail_text)
        for k in range(t+1, len(steps)):
            v = vals[(k-(t+1))] if (k-(t+1)) < len(vals) else ""
            later_pairs.append((k, v))
    for i in range(len(steps)):
        if i < t:
            executed_chunks_after.append(base_fixed_texts[i])
        elif i == t:
            executed_chunks_after.append(d_text)
        else:
            later = dict(later_pairs)
            ans_i = later.get(i, "")
            executed_chunks_after.append(f"(short rationale)\nANS: {ans_i}")

    # 6) Finalize (auto: attach_direct vs direct)
    f_final, f_cost = "", {"prompt_tokens":0,"completion_tokens":0,"latency_ms":0.0,"api_cost":0.0}
    if finalizer is not None:
        f_final, f_cost = run_finalize_auto(task, problem_block, steps, executed_chunks_after, finalizer)

    final = f_final if f_final else final_tail
    return d_text, d_cost, tail_cost, f_cost, final, tail_text, d_ans, executed_chunks_after

# ===== 데이터 스키마 =====
@dataclass
class ActionOutcome:
    action: str
    model: str
    step_output_text: str
    step_answer_only: str
    step_cost: Dict[str,float]
    step_docrel: Optional[float]=None
    final_answer_if_applied: Optional[str]=None
    final_correct_if_applied: Optional[bool]=None
    total_tokens_if_applied: Optional[int]=None
    total_latency_ms_if_applied: Optional[float]=None
    total_api_cost_if_applied: Optional[float]=None
    tail_steps: Optional[List[Tuple[int,str]]]=None
    tail_text_joined: Optional[str]=None
    format_ok: Optional[bool]=None
    ans_is_numeric: Optional[bool]=None
    retrieval_used: Optional[bool]=None
    llm_calls: Optional[int]=None
    total_tokens_label: Optional[int]=None
    total_api_cost_label: Optional[float]=None
    llm_calls_label: Optional[int]=None

@dataclass
class StepRecord:
    idx: int
    subquestion: str
    context: str
    actions: Dict[str,ActionOutcome]
    teacher_label: Optional[str]=None

@dataclass
class SampleRecord:
    id: str
    task: str
    query: Dict[str,Any]
    steps: List[StepRecord]
    baseline_final_answer: Optional[str]
    baseline_correct: Optional[bool]
    baseline_tot_tokens: Optional[int]
    baseline_tot_latency_ms: Optional[float]
    baseline_tot_api_cost: Optional[float]
    sample_gold: Optional[Dict[str,Any]]
    baseline_chain_text: Optional[str]=None
    baseline_two_line_steps: Optional[List[str]]=None
    baseline_ans_list: Optional[List[str]]=None

# ===== BM25 + 케이스 뱅크 =====
_WORD_RE = re.compile(r'[가-힣]+|[a-zA-Z0-9]+')
def _tok(text:str)->List[str]:
    return _WORD_RE.findall((text or "").lower())

class SimpleBM25:
    def __init__(self, docs:List[List[str]], k1:float=1.5, b:float=0.75):
        self.docs = docs
        self.N = len(docs)
        self.k1=k1; self.b=b
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len)/self.N) if self.N>0 else 0.0
        self.df: Dict[str,int] = {}
        for d in docs:
            for w in set(d):
                self.df[w] = self.df.get(w,0)+1
        self.idf: Dict[str,float] = {}
        for w,df in self.df.items():
            self.idf[w] = math.log((self.N - df + 0.5)/(df + 0.5) + 1.0)
    def score(self, q_tokens:List[str], idx:int)->float:
        d = self.docs[idx]
        dl = self.doc_len[idx] if idx < len(self.doc_len) else 0
        if dl==0 or self.avgdl==0: return 0.0
        tf: Dict[str,int] = {}
        for w in d:
            tf[w] = tf.get(w,0)+1
        s=0.0
        for w in q_tokens:
            if w not in tf or w not in self.idf: continue
            idf = self.idf[w]
            f = tf[w]
            denom = f + self.k1*(1 - self.b + self.b*dl/self.avgdl)
            s += idf * (f*(self.k1+1)) / (denom if denom>0 else 1e-9)
        return s
    def topk(self, q_text:str, k:int)->List[int]:
        q_tokens = _tok(q_text)
        scores = [(i, self.score(q_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i,_ in scores[:max(0,k)]]

def load_case_bank_jsonl(path:str)->List[Dict[str,Any]]:
    bank=[]
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln=ln.strip()
                if not ln: continue
                try:
                    obj=json.loads(ln)
                    if "subq" in obj and "resp" in obj:
                        bank.append(obj)
                except: continue
    except FileNotFoundError:
        pass
    return bank

def build_bm25_from_bank(bank:List[Dict[str,Any]])->SimpleBM25:
    docs = [_tok(str(rec.get("subq",""))) for rec in bank]
    return SimpleBM25(docs)

# ===== 메인 =====
def guess_qtype(q:str)->str:
    q=q.lower()
    if any(w in q for w in ["percent","percentage","increase","discount"]): return "percentage"
    if any(w in q for w in ["average","mean"]): return "average"
    if any(w in q for w in ["speed","rate","mph","km/h","distance","time"]): return "rate-time-distance"
    if any(w in q for w in ["rectangle","area","perimeter","square"]): return "geometry"
    return "unknown"

def ensure_dir(p:str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k","csqa"], default="gsm8k")
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--out", default="step0_train_unified.jsonl")
    ap.add_argument("--slm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--detract_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--escalate_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max_steps", type=int, default=3)
    ap.add_argument("--include_perstep_continue", action="store_true", default=True)
    ap.add_argument("--slm_dtype", default="float32")
    ap.add_argument("--llm_dtype", default="float16")
    ap.add_argument("--device_map", default="auto", help="transformers device_map (e.g., auto / cuda:0)")

    # Retrieval
    ap.add_argument("--ret_bank", default=None)
    ap.add_argument("--ret_topk", type=int, default=2)
    ap.add_argument("--ret_maxchars", type=int, default=400)
    ap.add_argument("--ret_disable_baseline", action="store_true")
    ap.add_argument("--bank_dump", default="bank_ret.jsonl")

    # 시드/덤프
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump_dir", default=None)

    # 무상 모델 등록
    ap.add_argument("--free_models", default=None)

    # 라벨 스코어
    ap.add_argument("--label_w_calls", type=float, default=1.0)
    ap.add_argument("--label_w_api", type=float, default=1.0)
    ap.add_argument("--label_w_tokens", type=float, default=0.001)
    ap.add_argument("--label_bias_detract", type=float, default=-0.05)
    ap.add_argument("--label_margin_eps", type=float, default=0.0)

    # CSQA
    ap.add_argument("--csqa_offset", type=int, default=0)
    ap.add_argument("--gsm_offset", type=int, default=0)

    # UX
    ap.add_argument("--no_tqdm", action="store_true", help="disable tqdm progress bar")
    ap.add_argument("--tqdm_force", action="store_true",
                help="show tqdm even if not a TTY (e.g., nohup/log file)")
    ap.add_argument("--reload_per_sample", action="store_true", help="(fallback) reload models per sample to save VRAM")

    args=ap.parse_args()
    set_global_seed(args.seed)

    # free model set
    FREE_MODELS.add(args.slm_model.lower())
    if args.free_models:
        for n in args.free_models.split(","):
            n=n.strip()
            if n: FREE_MODELS.add(n.lower())

    # Output dirs
    ensure_dir(os.path.dirname(args.out) or ".")
    if args.bank_dump: ensure_dir(os.path.dirname(args.bank_dump) or ".")

    # Dataset
    if args.task == "gsm8k":
        start = int(args.gsm_offset or 0)
        end = start + int(args.num_samples)
        base = load_dataset("gsm8k", "main")["train"]
        ds = base.select(range(start, min(end, len(base))))
        questions = ds["question"]; answers = ds["answer"]
        total_n = len(questions)
    else:
        hf = load_dataset("commonsense_qa")["train"]
        start = max(0, int(args.csqa_offset))
        end = start + int(args.num_samples) if args.num_samples > 0 else None
        hf_slice = hf.select(range(start, min(end, len(hf)) if end else len(hf)))
        csqa_list = []
        for j, ex in enumerate(hf_slice):
            csqa_list.append({
                "id": ex.get("id", f"csqa-{start+j:06d}"),
                "question": ex["question"],
                "choices": {"label": ex["choices"]["label"], "text": ex["choices"]["text"]},
                "answerKey": ex["answerKey"]
            })
        total_n = len(csqa_list)

    # Retrieval bank
    bank=[]; bm25=None
    if args.ret_bank:
        bank = load_case_bank_jsonl(args.ret_bank)
        if bank: bm25=build_bm25_from_bank(bank)

    # Model (pool or temp)
    slm_dtype = resolve_dtype_for_model(args.slm_model, args.slm_model, args.slm_dtype, args.llm_dtype)

    fout=open(args.out,"a",encoding="utf-8")
    ensure_dir(args.dump_dir) if args.dump_dir else None

    # label score helper
    def _label_score(v: ActionOutcome) -> float:
        calls = float(v.llm_calls_label or 0)
        api   = float(v.total_api_cost_label or 0.0)
        toks  = float(v.total_tokens_label or 0)
        bias  = (args.label_bias_detract if str(v.action or "").startswith("Detract@") else 0.0)
        return args.label_w_calls*calls + args.label_w_api*api + args.label_w_tokens*toks + bias

    # Progress bar
    force = args.tqdm_force or (os.environ.get("TQDM_FORCE", "0") == "1")
    disable_tqdm = args.no_tqdm or (not force and not sys.stderr.isatty())

    pbar = tqdm(
        total=total_n,
        desc=f"Chain pipeline [{args.task}]",
        dynamic_ncols=True,
        leave=True,
        disable=disable_tqdm,
        file=sys.stdout,          # 로그 파일로 합칠 때 보기 쉽게 stdout으로 고정
        mininterval=1.0,          # 파일/nohup 환경에서 과도한 업데이트 방지
    )
    # Prepare models (persistent) or set flags to reload per sample
    pool = None
    if not args.reload_per_sample:
        pool = ModelPool(device_map=args.device_map)
        slm = pool.get(args.slm_model, dtype=slm_dtype, seed=args.seed)

        detract_names=[m.strip() for m in args.detract_models.split(",") if m.strip()]
        escalate_names=[m.strip() for m in args.escalate_models.split(",") if m.strip()]
        finalizer_name = (escalate_names[0] if len(escalate_names)>0 else (detract_names[0] if len(detract_names)>0 else args.slm_model))

        detractors = []
        for mname in detract_names:
            d_dtype = resolve_dtype_for_model(mname, args.slm_model, args.slm_dtype, args.llm_dtype)
            detractors.append(pool.get(mname, dtype=d_dtype, seed=args.seed))
        escalators = []
        for mname in escalate_names:
            e_dtype = resolve_dtype_for_model(mname, args.slm_model, args.slm_dtype, args.llm_dtype)
            escalators.append(pool.get(mname, dtype=e_dtype, seed=args.seed))
        f_dtype = resolve_dtype_for_model(finalizer_name, args.slm_model, args.slm_dtype, args.llm_dtype)
        finalizer = pool.get(finalizer_name, dtype=f_dtype, seed=args.seed)
    else:
        # Fallback: re-create per sample to save VRAM
        slm = HFChatCaller(args.slm_model, dtype=slm_dtype, seed=args.seed, device_map=args.device_map)
        detract_names=[m.strip() for m in args.detract_models.split(",") if m.strip()]
        escalate_names=[m.strip() for m in args.escalate_models.split(",") if m.strip()]
        finalizer_name = (escalate_names[0] if len(escalate_names)>0 else (detract_names[0] if len(detract_names)>0 else args.slm_model))

    try:
        for i in range(total_n):
            try:
                if args.task=="gsm8k":
                    pid = f"gsm8k-{(int(args.gsm_offset or 0) + i):06d}"
                    prob=questions[i].strip()
                    gold= last_number(answers[i]) or answers[i].strip()
                    qtype=guess_qtype(prob)

                    # 1) Decompose
                    deco_text, _ = slm.chat(prompt_decompose_math(prob, qtype), max_new_tokens=200, temperature=0.0)
                    steps=parse_decompose(deco_text, ["Compute the next needed count."]*3)
                    steps=steps[:args.max_steps] if args.max_steps else steps
                    if len(steps)<3: steps=(steps+["Compute the next needed count."]*3)[:3]

                    # 2) Baseline chain
                    base_ans_list, base_final, base_cost, base_text = run_singlepass_chain_math(slm, prob, steps)
                    base_ok = judge_numeric(gold, base_final or "")

                    # 2-1) 두줄 리포맷
                    base_fixed_texts=[]
                    if any((a or "").strip()=="" for a in base_ans_list):
                        fixed_blocks, _, _ = reformat_two_line_steps(slm, prob, steps, base_text, last_is_letter=False)
                        base_fixed_texts = fixed_blocks if len(fixed_blocks)==3 else [
                            f"(short rationale)\nANS: {base_ans_list[j] if j<len(base_ans_list) else ''}" for j in range(3)
                        ]
                    else:
                        for j in range(3):
                            a = base_ans_list[j] if j<len(base_ans_list) else ""
                            base_fixed_texts.append(f"(short rationale)\nANS: {a}")

                    problem_block = prob

                    # 스텝 레코드/Continue
                    step_records=[StepRecord(idx=t, subquestion=steps[t], context=f"[MATH] {qtype}", actions={}) for t in range(3)]
                    if args.include_perstep_continue:
                        for t in range(3):
                            ans_val = extract_last_ans_line(base_fixed_texts[t])
                            isnum = bool(re.fullmatch(r'-?\d+(?:\.\d+)?', (ans_val or "").strip()))
                            step_records[t].actions["Continue@SLM"]=ActionOutcome(
                                action="Continue@SLM", model=slm.model_name,
                                step_output_text=base_fixed_texts[t], step_answer_only=ans_val,
                                step_cost={"prompt_tokens":0,"completion_tokens":0,"latency_ms":0.0,"api_cost":0.0},
                                final_answer_if_applied=base_final, final_correct_if_applied=bool(base_ok),
                                total_tokens_if_applied=int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                                total_latency_ms_if_applied=float(base_cost["latency_ms"]),
                                total_api_cost_if_applied=float(base_cost["api_cost"]),
                                tail_steps=[(j, extract_last_ans_line(base_fixed_texts[j])) for j in range(t+1,3)],
                                tail_text_joined=base_text[:1200],
                                format_ok=True, ans_is_numeric=isnum, retrieval_used=False,
                                llm_calls=0, total_tokens_label=int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                                total_api_cost_label=float(base_cost["api_cost"]), llm_calls_label=0
                            )

                    # 모델들 준비 (reload_per_sample인 경우에만 이 시점에서 로딩)
                    if args.reload_per_sample:
                        detractors=[]; escalators=[]
                        for mname in detract_names:
                            d_dtype = resolve_dtype_for_model(mname, args.slm_model, args.slm_dtype, args.llm_dtype)
                            detractors.append(HFChatCaller(mname, dtype=d_dtype, seed=args.seed, device_map=args.device_map))
                        for mname in escalate_names:
                            e_dtype = resolve_dtype_for_model(mname, args.slm_model, args.slm_dtype, args.llm_dtype)
                            escalators.append(HFChatCaller(mname, dtype=e_dtype, seed=args.seed, device_map=args.device_map))
                        f_dtype = resolve_dtype_for_model(finalizer_name, args.slm_model, args.slm_dtype, args.llm_dtype)
                        finalizer = HFChatCaller(finalizer_name, dtype=f_dtype, seed=args.seed, device_map=args.device_map)

                    # Detract
                    for dcaller in detractors if not args.reload_per_sample else detractors:
                        mname = dcaller.model_name
                        try:
                            for t in range(3):
                                support_pairs=[]
                                if bm25 and bank:
                                    idxs=bm25.topk(steps[t], max(1,args.ret_topk))
                                    for k in idxs:
                                        rec=bank[k]; subq=str(rec.get("subq","")).strip(); resp=str(rec.get("resp","")).strip()
                                        if args.ret_maxchars and len(resp)>args.ret_maxchars: resp=resp[:args.ret_maxchars].rstrip()+" ..."
                                        support_pairs.append((subq,resp))
                                d_text, d_cost, tail_cost, f_cost, final, tail_text, d_ans, _ = run_detract_then_slm_with_finalize(
                                    "gsm8k", slm, dcaller, finalizer, problem_block, steps, base_fixed_texts, t, support_pairs=support_pairs
                                )
                                tot_tok = d_cost["prompt_tokens"]+d_cost["completion_tokens"] + tail_cost["prompt_tokens"]+tail_cost["completion_tokens"] + (f_cost.get("prompt_tokens",0)+f_cost.get("completion_tokens",0))
                                tot_api = d_cost.get("api_cost",0.0)+tail_cost.get("api_cost",0.0)+f_cost.get("api_cost",0.0)
                                tot_lat = d_cost.get("latency_ms",0.0)+tail_cost.get("latency_ms",0.0)+f_cost.get("latency_ms",0.0)
                                ok = judge_numeric(gold, final or "")
                                lab_tok = d_cost["prompt_tokens"]+d_cost["completion_tokens"] + tail_cost["prompt_tokens"]+tail_cost["completion_tokens"]
                                lab_api = d_cost.get("api_cost",0.0)+tail_cost.get("api_cost",0.0)
                                lab_calls = 1
                                isnum = bool(re.fullmatch(r'-?\d+(?:\.\d+)?', (d_ans or "").strip()))
                                analysis_calls = 1 + (1 if (t == 2) else 0) + (1 if f_cost.get("prompt_tokens",0)>0 else 0)
                                step_records[t].actions[f"Detract@{mname}"]=ActionOutcome(
                                    action=f"Detract@{mname}", model=mname,
                                    step_output_text=d_text, step_answer_only=d_ans, step_cost=d_cost,
                                    final_answer_if_applied=final, final_correct_if_applied=bool(ok),
                                    total_tokens_if_applied=int(tot_tok), total_latency_ms_if_applied=float(tot_lat),
                                    total_api_cost_if_applied=float(tot_api), tail_steps=None,
                                    tail_text_joined=tail_text[:1200], retrieval_used=bool(support_pairs),
                                    ans_is_numeric=isnum, llm_calls=analysis_calls,
                                    total_tokens_label=int(lab_tok), total_api_cost_label=float(lab_api), llm_calls_label=lab_calls
                                )
                        except Exception as e:
                            tqdm.write(f"[WARN] detract failed ({mname}): {e}")

                    # Escalate
                    def run_escalate_once(ecaller:HFChatCaller, t:int):
                        solved_prefix=[(j, base_fixed_texts[j]) for j in range(t)]
                        msgs = build_chain_prompt_strict_math(prob, steps, solved_prefix=solved_prefix, start_idx=t, include_step_t=False, step_t_text=None)
                        text, cost = ecaller.chat(msgs, max_new_tokens=650, temperature=0.0)
                        final = extract_final_number(text)
                        first_ans = extract_first_ans_line(text)
                        return text, cost, final, first_ans

                    for ecaller in escalators if not args.reload_per_sample else escalators:
                        mname = ecaller.model_name
                        try:
                            for t in range(3):
                                text, cost, final, first_ans = run_escalate_once(ecaller, t)
                                ok = judge_numeric(gold, final or "")
                                isnum = bool(re.fullmatch(r'-?\d+(?:\.\d+)?', (first_ans or "").strip()))
                                step_records[t].actions[f"Escalate@{mname}"]=ActionOutcome(
                                    action=f"Escalate@{mname}", model=mname,
                                    step_output_text=f"(generated from step {t+1})\nANS: {first_ans}",
                                    step_answer_only=first_ans, step_cost=cost,
                                    final_answer_if_applied=final, final_correct_if_applied=bool(ok),
                                    total_tokens_if_applied=int(cost["prompt_tokens"]+cost["completion_tokens"]),
                                    total_latency_ms_if_applied=float(cost["latency_ms"]),
                                    total_api_cost_if_applied=float(cost["api_cost"]),
                                    tail_steps=None, tail_text_joined=text[:1200],
                                    ans_is_numeric=isnum, llm_calls=1,
                                    total_tokens_label=int(cost["prompt_tokens"]+cost["completion_tokens"]),
                                    total_api_cost_label=float(cost["api_cost"]),
                                    llm_calls_label=1
                                )
                        except Exception as e:
                            tqdm.write(f"[WARN] escalate failed ({mname}): {e}")

                    # 4) 라벨 선정
                    for st in step_records:
                        acts = st.actions
                        cont = acts.get("Continue@SLM")
                        need_fix = bool(cont and not cont.ans_is_numeric)  # math: 숫자 필요
                        cand_items = list(acts.items())
                        if need_fix:
                            def _produces_numeric(v: ActionOutcome)->bool:
                                if v.ans_is_numeric is not None: return bool(v.ans_is_numeric)
                                a = v.step_answer_only or extract_last_ans_line(v.step_output_text)
                                return bool(re.fullmatch(r'-?\d+(?:\.\d+)?', (a or "").strip()))
                            fixers = [(k,v) for k,v in cand_items if _produces_numeric(v)]
                            if fixers: cand_items = fixers
                        corrects = [(k,v) for k,v in cand_items if v.final_correct_if_applied]
                        pool_cands = corrects if corrects else cand_items
                        scored = [(k,v,_label_score(v)) for (k,v) in pool_cands]
                        scored.sort(key=lambda x: x[2])
                        st.teacher_label = scored[0][0] if scored else None

                    # 5) bank_dump
                    if args.bank_dump:
                        seen=set()
                        with open(args.bank_dump,"a",encoding="utf-8") as fb:
                            for st in step_records:
                                chosen = st.actions.get(st.teacher_label) if st.teacher_label else None
                                resp_text = (chosen.step_output_text if (chosen and chosen.step_output_text) else base_fixed_texts[st.idx])
                                ans_val = extract_last_ans_line(resp_text) or (chosen.step_answer_only if chosen else extract_last_ans_line(base_fixed_texts[st.idx]))
                                item = make_bank_item(pid, st.idx, st.subquestion, resp_text, ans_val or "", st.teacher_label or "Continue@SLM", (chosen.model if chosen else slm.model_name))
                                if item["hash_key"] not in seen:
                                    fb.write(json.dumps(item, ensure_ascii=False)+"\n"); seen.add(item["hash_key"])

                    # 6) sample_gold
                    def _key_label_cont(d):
                        calls=float(d.get("llm_calls_label",99)); api=float(d.get("total_api_cost_label",1e9)); toks=float(d.get("total_tokens_label",1e9))
                        bias = args.label_bias_detract if str(d.get("action","")).startswith("Detract@") else 0.0
                        return args.label_w_calls*calls + args.label_w_api*api + args.label_w_tokens*toks + bias

                    best=None
                    if base_ok:
                        best={"step_idx": None, "action":"Continue@SLM", "model": slm.model_name,
                              "tot_api_cost": float(base_cost["api_cost"]), "tot_tokens": int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                              "tot_latency_ms": float(base_cost["latency_ms"]), "final_answer": base_final,
                              "llm_calls_label": 0, "total_api_cost_label": float(base_cost["api_cost"]), "total_tokens_label": int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]), "llm_calls":0}
                    else:
                        pool_c=[]
                        for st in step_records:
                            for a in st.actions.values():
                                if a.final_correct_if_applied:
                                    pool_c.append({"step_idx": st.idx, "action": a.action, "model": a.model,
                                                   "final_answer": a.final_answer_if_applied,
                                                   "llm_calls_label": a.llm_calls_label,
                                                   "total_api_cost_label": a.total_api_cost_label,
                                                   "total_tokens_label": a.total_tokens_label,
                                                   "llm_calls": a.llm_calls,
                                                   "tot_api_cost": a.total_api_cost_if_applied,
                                                   "tot_tokens": a.total_tokens_if_applied,
                                                   "tot_latency_ms": a.total_latency_ms_if_applied})
                        if pool_c: best=min(pool_c, key=_key_label_cont)
                    if best is None:
                        pool_c=[{"step_idx": None,"action":"Continue@SLM","model":slm.model_name,"final_answer":base_final,
                               "llm_calls_label":0,"total_api_cost_label":float(base_cost["api_cost"]),"total_tokens_label":int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                               "llm_calls":0,"tot_api_cost":float(base_cost["api_cost"]),"tot_tokens":int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                               "tot_latency_ms":float(base_cost["latency_ms"])}]
                        for st in step_records:
                            for a in st.actions.values():
                                pool_c.append({"step_idx": st.idx, "action": a.action, "model": a.model,
                                               "final_answer": a.final_answer_if_applied,
                                               "llm_calls_label": a.llm_calls_label,
                                               "total_api_cost_label": a.total_api_cost_label,
                                               "total_tokens_label": a.total_tokens_label,
                                               "llm_calls": a.llm_calls,
                                               "tot_api_cost": a.total_api_cost_if_applied,
                                               "tot_tokens": a.total_tokens_if_applied,
                                               "tot_latency_ms": a.total_latency_ms_if_applied})
                        best=min(pool_c, key=_key_label_cont)

                    rec=SampleRecord(
                        id=pid, task="math",
                        query={"problem": prob, "type": qtype, "solution": str(gold)},
                        steps=step_records,
                        baseline_final_answer=base_final, baseline_correct=bool(base_ok),
                        baseline_tot_tokens=int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                        baseline_tot_latency_ms=float(base_cost["latency_ms"]),
                        baseline_tot_api_cost=float(base_cost["api_cost"]),
                        sample_gold=best,
                        baseline_chain_text=base_text,
                        baseline_two_line_steps=base_fixed_texts,
                        baseline_ans_list=base_ans_list
                    )
                    fout.write(json.dumps(asdict(rec), ensure_ascii=False)+"\n"); fout.flush()

                    # unload per-sample models if using reload_per_sample
                    if args.reload_per_sample:
                        for c in detractors + escalators + [finalizer]:
                            try: c.close()
                            except: pass
                        detractors.clear(); escalators.clear()

                else:
                    # ===== CSQA pipeline =====
                    obj = csqa_list[i]
                    pid = obj.get("id") or f"csqa-{i:06d}"
                    q = obj["question"].strip()
                    labels = obj["choices"]["label"]; texts = obj["choices"]["text"]
                    gold_letter = (obj.get("answerKey","") or "").strip().upper()
                    problem_block = format_mcqa_problem(q, labels, texts)

                    deco_text, _ = slm.chat(prompt_decompose_csqa(q, "\n".join([f"{l}. {t}" for l,t in zip(labels,texts)])), max_new_tokens=220, temperature=0.0)
                    steps = parse_decompose(deco_text, ["Analyze the question context.","Filter the plausible options.","Choose the best option."])
                    steps=steps[:args.max_steps] if args.max_steps else steps
                    if len(steps)<3: steps=(steps+["Refine the options."]*3)[:3]

                    base_ans_list, base_final, base_cost, base_text = run_singlepass_chain_csqa(slm, problem_block, steps)
                    base_ok = judge_letter(gold_letter, base_final or "")

                    base_fixed_texts=[]
                    if any((a or "").strip()=="" for a in base_ans_list):
                        fixed_blocks, _, _ = reformat_two_line_steps(slm, problem_block, steps, base_text, last_is_letter=True)
                        base_fixed_texts = fixed_blocks if len(fixed_blocks)==3 else [
                            f"(short rationale)\nANS: {base_ans_list[j] if j<len(base_ans_list) else ''}" for j in range(3)
                        ]
                    else:
                        for j in range(3):
                            a = base_ans_list[j] if j<len(base_ans_list) else ""
                            base_fixed_texts.append(f"(short rationale)\nANS: {a}")

                    step_records=[StepRecord(idx=t, subquestion=steps[t], context="[CSQA]", actions={}) for t in range(3)]
                    if args.include_perstep_continue:
                        for t in range(3):
                            ans_val = extract_last_ans_line(base_fixed_texts[t])
                            is_valid = bool((ans_val or "").strip())
                            step_records[t].actions["Continue@SLM"]=ActionOutcome(
                                action="Continue@SLM", model=slm.model_name,
                                step_output_text=base_fixed_texts[t], step_answer_only=ans_val,
                                step_cost={"prompt_tokens":0,"completion_tokens":0,"latency_ms":0.0,"api_cost":0.0},
                                final_answer_if_applied=base_final, final_correct_if_applied=bool(base_ok),
                                total_tokens_if_applied=int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                                total_latency_ms_if_applied=float(base_cost["latency_ms"]),
                                total_api_cost_if_applied=float(base_cost["api_cost"]),
                                tail_steps=[(j, extract_last_ans_line(base_fixed_texts[j])) for j in range(t+1,3)],
                                tail_text_joined=base_text[:1200],
                                format_ok=True, ans_is_numeric=None, retrieval_used=False,
                                llm_calls=0, total_tokens_label=int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                                total_api_cost_label=float(base_cost["api_cost"]), llm_calls_label=0
                            )

                    if args.reload_per_sample:
                        detractors=[]; escalators=[]
                        for mname in detract_names:
                            d_dtype = resolve_dtype_for_model(mname, args.slm_model, args.slm_dtype, args.llm_dtype)
                            detractors.append(HFChatCaller(mname, dtype=d_dtype, seed=args.seed, device_map=args.device_map))
                        for mname in escalate_names:
                            e_dtype = resolve_dtype_for_model(mname, args.slm_model, args.slm_dtype, args.llm_dtype)
                            escalators.append(HFChatCaller(mname, dtype=e_dtype, seed=args.seed, device_map=args.device_map))
                        f_dtype = resolve_dtype_for_model(finalizer_name, args.slm_model, args.slm_dtype, args.llm_dtype)
                        finalizer = HFChatCaller(finalizer_name, dtype=f_dtype, seed=args.seed, device_map=args.device_map)

                    for dcaller in detractors if not args.reload_per_sample else detractors:
                        mname = dcaller.model_name
                        try:
                            for t in range(3):
                                support_pairs=[]
                                if bm25 and bank:
                                    idxs=bm25.topk(steps[t], max(1,args.ret_topk))
                                    for k in idxs:
                                        rec=bank[k]; subq=str(rec.get("subq","")).strip(); resp=str(rec.get("resp","")).strip()
                                        if args.ret_maxchars and len(resp)>args.ret_maxchars: resp=resp[:args.ret_maxchars].rstrip()+" ..."
                                        support_pairs.append((subq,resp))
                                d_text, d_cost, tail_cost, f_cost, final, tail_text, d_ans, _ = run_detract_then_slm_with_finalize(
                                    "csqa", slm, dcaller, finalizer, problem_block, steps, base_fixed_texts, t, support_pairs=support_pairs
                                )
                                tot_tok = d_cost["prompt_tokens"]+d_cost["completion_tokens"] + tail_cost["prompt_tokens"]+tail_cost["completion_tokens"] + (f_cost.get("prompt_tokens",0)+f_cost.get("completion_tokens",0))
                                tot_api = d_cost.get("api_cost",0.0)+tail_cost.get("api_cost",0.0)+f_cost.get("api_cost",0.0)
                                tot_lat = d_cost.get("latency_ms",0.0)+tail_cost.get("latency_ms",0.0)+f_cost.get("latency_ms",0.0)
                                ok = judge_letter(gold_letter, final or "")
                                lab_tok = d_cost["prompt_tokens"]+d_cost["completion_tokens"] + tail_cost["prompt_tokens"]+tail_cost["completion_tokens"]
                                lab_api = d_cost.get("api_cost",0.0)+tail_cost.get("api_cost",0.0)
                                lab_calls = 1
                                is_valid = bool((d_ans or "").strip())
                                analysis_calls = 1 + (1 if (t == 2) else 0) + (1 if f_cost.get("prompt_tokens",0)>0 else 0)
                                step_records[t].actions[f"Detract@{mname}"]=ActionOutcome(
                                    action=f"Detract@{mname}", model=mname,
                                    step_output_text=d_text, step_answer_only=d_ans, step_cost=d_cost,
                                    final_answer_if_applied=final, final_correct_if_applied=bool(ok),
                                    total_tokens_if_applied=int(tot_tok), total_latency_ms_if_applied=float(tot_lat),
                                    total_api_cost_if_applied=float(tot_api), tail_steps=None,
                                    tail_text_joined=tail_text[:1200], retrieval_used=bool(support_pairs),
                                    ans_is_numeric=None, llm_calls=analysis_calls,
                                    total_tokens_label=int(lab_tok), total_api_cost_label=float(lab_api), llm_calls_label=lab_calls
                                )
                        except Exception as e:
                            tqdm.write(f"[WARN] detract failed ({mname}): {e}")

                    def run_escalate_once_csqa(ecaller:HFChatCaller, t:int):
                        solved_prefix=[(j, base_fixed_texts[j]) for j in range(t)]
                        msgs = build_chain_prompt_strict_csqa(problem_block, steps, solved_prefix=solved_prefix, start_idx=t, include_step_t=False, step_t_text=None)
                        text, cost = ecaller.chat(msgs, max_new_tokens=650, temperature=0.0)
                        final = extract_final_letter(text, labels)
                        first_ans = extract_first_ans_line(text)
                        return text, cost, final, first_ans

                    for ecaller in escalators if not args.reload_per_sample else escalators:
                        mname = ecaller.model_name
                        try:
                            for t in range(3):
                                text, cost, final, first_ans = run_escalate_once_csqa(ecaller, t)
                                ok = judge_letter(gold_letter, final or "")
                                is_valid = bool((first_ans or "").strip())
                                step_records[t].actions[f"Escalate@{mname}"]=ActionOutcome(
                                    action=f"Escalate@{mname}", model=mname,
                                    step_output_text=f"(generated from step {t+1})\nANS: {first_ans}",
                                    step_answer_only=first_ans, step_cost=cost,
                                    final_answer_if_applied=final, final_correct_if_applied=bool(ok),
                                    total_tokens_if_applied=int(cost["prompt_tokens"]+cost["completion_tokens"]),
                                    total_latency_ms_if_applied=float(cost["latency_ms"]),
                                    total_api_cost_if_applied=float(cost["api_cost"]),
                                    tail_steps=None, tail_text_joined=text[:1200],
                                    ans_is_numeric=None, llm_calls=1,
                                    total_tokens_label=int(cost["prompt_tokens"]+cost["completion_tokens"]),
                                    total_api_cost_label=float(cost["api_cost"]),
                                    llm_calls_label=1
                                )
                        except Exception as e:
                            tqdm.write(f"[WARN] escalate failed ({mname}): {e}")

                    for st in step_records:
                        acts = st.actions
                        cont = acts.get("Continue@SLM")
                        need_fix = bool(cont and not (cont.step_answer_only or "").strip())
                        cand_items = list(acts.items())
                        if need_fix:
                            def _produces_nonempty(v: ActionOutcome)->bool:
                                a = v.step_answer_only or extract_last_ans_line(v.step_output_text)
                                return bool((a or "").strip())
                            fixers = [(k,v) for k,v in cand_items if _produces_nonempty(v)]
                            if fixers: cand_items = fixers
                        corrects = [(k,v) for k,v in cand_items if v.final_correct_if_applied]
                        pool_cands = corrects if corrects else cand_items
                        scored = [(k,v,_label_score(v)) for (k,v) in pool_cands]
                        scored.sort(key=lambda x: x[2])
                        st.teacher_label = scored[0][0] if scored else None

                    if args.bank_dump:
                        seen=set()
                        with open(args.bank_dump,"a",encoding="utf-8") as fb:
                            for st in step_records:
                                chosen = st.actions.get(st.teacher_label) if st.teacher_label else None
                                resp_text = (chosen.step_output_text if (chosen and chosen.step_output_text) else base_fixed_texts[st.idx])
                                ans_val = extract_last_ans_line(resp_text) or (chosen.step_answer_only if chosen else extract_last_ans_line(base_fixed_texts[st.idx]))
                                item = make_bank_item(pid, st.idx, st.subquestion, resp_text, ans_val or "", st.teacher_label or "Continue@SLM", (chosen.model if chosen else slm.model_name))
                                if item["hash_key"] not in seen:
                                    fb.write(json.dumps(item, ensure_ascii=False)+"\n"); seen.add(item["hash_key"])

                    def _key_label_cont(d):
                        calls=float(d.get("llm_calls_label",99)); api=float(d.get("total_api_cost_label",1e9)); toks=float(d.get("total_tokens_label",1e9))
                        bias = args.label_bias_detract if str(d.get("action","")).startswith("Detract@") else 0.0
                        return args.label_w_calls*calls + args.label_w_api*api + args.label_w_tokens*toks + bias

                    best=None
                    if base_ok:
                        best={"step_idx": None, "action":"Continue@SLM", "model": slm.model_name,
                              "tot_api_cost": float(base_cost["api_cost"]), "tot_tokens": int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                              "tot_latency_ms": float(base_cost["latency_ms"]), "final_answer": base_final,
                              "llm_calls_label": 0, "total_api_cost_label": float(base_cost["api_cost"]), "total_tokens_label": int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]), "llm_calls":0}
                    else:
                        pool_c=[]
                        for st in step_records:
                            for a in st.actions.values():
                                if a.final_correct_if_applied:
                                    pool_c.append({"step_idx": st.idx, "action": a.action, "model": a.model,
                                                   "final_answer": a.final_answer_if_applied,
                                                   "llm_calls_label": a.llm_calls_label,
                                                   "total_api_cost_label": a.total_api_cost_label,
                                                   "total_tokens_label": a.total_tokens_label,
                                                   "llm_calls": a.llm_calls,
                                                   "tot_api_cost": a.total_api_cost_if_applied,
                                                   "tot_tokens": a.total_tokens_if_applied,
                                                   "tot_latency_ms": a.total_latency_ms_if_applied})
                        if pool_c: best=min(pool_c, key=_key_label_cont)
                    if best is None:
                        pool_c=[{"step_idx": None,"action":"Continue@SLM","model":slm.model_name,"final_answer":base_final,
                               "llm_calls_label":0,"total_api_cost_label":float(base_cost["api_cost"]),"total_tokens_label":int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                               "llm_calls":0,"tot_api_cost":float(base_cost["api_cost"]),"tot_tokens":int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                               "tot_latency_ms":float(base_cost["latency_ms"])}]
                        for st in step_records:
                            for a in st.actions.values():
                                pool_c.append({"step_idx": st.idx, "action": a.action, "model": a.model,
                                               "final_answer": a.final_answer_if_applied,
                                               "llm_calls_label": a.llm_calls_label,
                                               "total_api_cost_label": a.total_api_cost_label,
                                               "total_tokens_label": a.total_tokens_label,
                                               "llm_calls": a.llm_calls,
                                               "tot_api_cost": a.total_api_cost_if_applied,
                                               "tot_tokens": a.total_tokens_if_applied,
                                               "tot_latency_ms": a.total_latency_ms_if_applied})
                        best=min(pool_c, key=_key_label_cont)

                    rec=SampleRecord(
                        id=pid, task="mcqa",
                        query={"question": q, "choices": [{"label":l,"text":t} for l,t in zip(labels,texts)], "gold": gold_letter},
                        steps=step_records,
                        baseline_final_answer=base_final, baseline_correct=bool(base_ok),
                        baseline_tot_tokens=int(base_cost["prompt_tokens"]+base_cost["completion_tokens"]),
                        baseline_tot_latency_ms=float(base_cost["latency_ms"]),
                        baseline_tot_api_cost=float(base_cost["api_cost"]),
                        sample_gold=best,
                        baseline_chain_text=base_text,
                        baseline_two_line_steps=base_fixed_texts,
                        baseline_ans_list=base_ans_list
                    )
                    fout.write(json.dumps(asdict(rec), ensure_ascii=False)+"\n"); fout.flush()

                # progress
                pbar.update(1)

            except KeyboardInterrupt:
                tqdm.write("\n[INTERRUPT] Stopping gracefully...")
                break
            except Exception as e:
                tqdm.write(f"[ERROR] sample {i}: {e}\n{traceback.format_exc()}")
                pbar.update(1)
                continue

    finally:
        pbar.close()
        fout.close()
        try:
            if pool: pool.close_all()
            else: slm.close()
        except: pass
        gc.collect(); torch.cuda.empty_cache()

if __name__=="__main__":
    main()
