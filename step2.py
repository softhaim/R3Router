#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Router-driven inference on GSM8K/CSQA (test/validation) — compact build with step-level debug prints

변경 요약:
- [비용] Router/Random에도 SLM decompose + SLM chain 비용을 공정하게 합산 (--charge_slm_to_router)
- [지표] 성능/비용 증가비, Cost Efficiency, IRT-style reward, TOPSIS 계산/출력
- [실행] 모델 리다이렉트 (--redirect_models "src|dst[,src2|dst2]"), 예: CSQA에서 14B→7B
"""

import os, re, json, math, random, argparse, gc, time
os.environ["CUDA_VISIBLE_DEVICES"]=os.environ.get("CUDA_VISIBLE_DEVICES","2,3")
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from datasets import load_dataset
from tqdm import tqdm

ACTIONS = ["Continue", "Detract", "Escalate"]

# =========[ DEBUG PRINT (compact) ]=========
def _truncate(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n].rstrip() + " ...")

def print_case_header(i, q, gold):
    print("\n" + "="*90)
    print(f"[Case #{i}] GOLD={gold}")
    print("- Problem -------------------------------")
    print(q)

def print_slm_debug(steps, base_text, parsed):
    print("- SLM steps -----------------------------")
    for k, s in enumerate(steps, 1):
        print(f"  Step {k}. {s}")
    print("- SLM raw chain -------------------------")
    print(_truncate(base_text, 800))
    print("- Parsed ANS per step -------------------")
    for i, pi in enumerate(parsed, 1):
        print(f"  [{i}] has={pi['has']} num={pi['isnum']} ANS={pi['value']}")

def print_decision_step(prefix, t, action, model, *, q_step, slm_chunk_before, edited_text=None, info=None):
    print(f"- [{prefix}] step {t+1}: action={action} model={model}")
    if info: print("  info:", json.dumps(info, ensure_ascii=False))
    print(f"  [Q] {q_step}")
    if slm_chunk_before:
        one = slm_chunk_before.strip().splitlines()[:2]
        for ln in one:
            print("  [A_SLM]", _truncate(ln, 160))
    if edited_text:
        print("  edited:")
        for ln in edited_text.strip().splitlines()[:2]:
            print("   ", _truncate(ln, 160))

def print_result_multi(base_final, base_ok, llm_results_ordered, router_final, router_ok, random_final, random_ok):
    print("- Results --------------------------------")
    print(f"Baseline FINAL={base_final}  -> {'OK' if base_ok else 'WRONG'}")
    for m, r in llm_results_ordered:
        print(f"LLM-only[{m}] FINAL={r['final']}  -> {'OK' if r['correct'] else 'WRONG'}")
    print(f"Router   FINAL={router_final} -> {'OK' if router_ok else 'WRONG'}")
    print(f"Random   FINAL={random_final} -> {'OK' if random_ok else 'WRONG'}")
# ===========================================

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ---- number utils ----
NUM_WITH_COMMAS = re.compile(r'-?\d[\d,]*(?:\.\d+)?')

def last_number(text: str) -> Optional[str]:
    if not text: return None
    toks = NUM_WITH_COMMAS.findall(text)
    if not toks: return None
    for tok in reversed(toks):
        t = tok.replace(",", "")
        if len(t) >= 1: return t
    return toks[-1].replace(",", "")

def extract_final_line(text: str) -> Optional[str]:
    if not text: return None
    for p in [r'^\s*FINAL(?:\s*ANSWER)?\s*[:=]\s*(.+?)\s*$',
              r'^\s*Answer\s*[:=]\s*(.+?)\s*$',
              r'^\s*Result\s*[:=]\s*(.+?)\s*$']:
        m = re.search(p, text, flags=re.I|re.M)
        if m: return m.group(1).strip()
    return None

def judge_numeric(gold: str, pred: str) -> bool:
    def _pick(s):
        m = NUM_WITH_COMMAS.findall(str(s))
        return m[-1].replace(",","") if m else None
    g, p = _pick(gold), _pick(pred)
    return (g==p) if (g is not None and p is not None) else (str(gold).strip()==str(pred).strip())

# =========[ Task switch & CSQA helpers ]=========
TASK = "gsm8k"  # or "csqa"

def set_task(t: str):
    global TASK
    TASK = (t or "gsm8k").lower()

def is_csqa() -> bool:
    # MCQA 전용 분기: csqa/openbookqa/race/race-middle/race-high 은 전부 객관식 처리
    return TASK in {"csqa", "openbookqa", "race", "race-middle", "race-high"}

CHOICE_LETTER_RE = re.compile(r'\b([A-J])\b', flags=re.I)

def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = CHOICE_LETTER_RE.search(text)
    return m.group(1).upper() if m else None

def judge_answer(gold: str, pred: str) -> bool:
    if is_csqa():
        g = (gold or "").strip().upper()[:1]
        p = (pred or "").strip()
        p = extract_choice_letter(p) or p[:1].upper()
        return bool(g) and (p == g)
    else:
        return judge_numeric(gold, pred)

# ---- cost model ----
MODEL_COST_CONFIG = {
    'qwen25_1p5b_instruct':  {'input_cost': 0.05e-6,  'output_cost': 0.10e-6},
    'qwen25_7b_instruct':    {'input_cost': 0.10e-6,  'output_cost': 0.20e-6},
    'qwen25_14b_instruct':   {'input_cost': 0.15e-6,  'output_cost': 0.30e-6},
    'llama31_8b_instruct':   {'input_cost': 0.10e-6,  'output_cost': 0.20e-6},
}
FREE_MODELS = set()
def is_free_model(n:str)->bool: return n.lower() in FREE_MODELS
def cost_key_from_model_name(n:str)->Optional[str]:
    n=n.lower()
    if "qwen" in n and ("2.5" in n or "2-5" in n or "2_5" in n):
        if "1.5b" in n or "1_5b" in n: return "qwen25_1p5b_instruct"
        if "7b"   in n: return "qwen25_7b_instruct"
        if "14b"  in n: return "qwen25_14b_instruct"
    if "llama-3.1" in n or "llama3.1" in n or "llama-3_1" in n:
        if "8b" in n: return "llama31_8b_instruct"
    return None
def apply_api_cost(cost: Dict[str,float], model_name: str)->None:
    if is_free_model(model_name): cost['api_cost']=0.0; return
    key = cost_key_from_model_name(model_name)
    if key and key in MODEL_COST_CONFIG:
        ic, oc = MODEL_COST_CONFIG[key]['input_cost'], MODEL_COST_CONFIG[key]['output_cost']
        pt, ct = int(cost.get("prompt_tokens",0)), int(cost.get("completion_tokens",0))
        cost['api_cost'] = float(ic*pt + oc*ct)
    else: cost['api_cost']=0.0

# ---- dtype force ----
FORCE_DTYPE: Dict[str, str] = {}
def register_fp32_models(fp32_list: str):
    FORCE_DTYPE.clear()
    for m in (fp32_list or "").split(","):
        mm = m.strip()
        if mm: FORCE_DTYPE[mm.lower()] = "float32"
def resolve_dtype(model_name: str, default: str = "float16") -> str:
    return FORCE_DTYPE.get(model_name.lower(), default)

# === [ADD] SLM 체인 결과를 스텝별 '가상 비용'으로 분할 ===
def _count_tokens_tok(tok, text: str) -> int:
    if not text:
        return 0
    try:
        enc = tok(text, return_tensors="pt")
        return int(enc["input_ids"].numel())
    except Exception:
        try:
            return int(len(tok.encode(text)))
        except Exception:
            return max(0, len(text) // 2)  # 아주 드문 예외 대비

def build_virtual_slm_step_costs(base_parsed, base_cost, slm_model_name: str, slm_tokenizer, prompt_share_mode: str = "equal"):
    """
    base_parsed: run_singlepass_chain()이 반환한 parsed (각 step의 chunk 포함)
    base_cost  : run_singlepass_chain()의 전체 SLM 체인 비용 (prompt/completion 합계)
    slm_model_name: SLM 모델명 (비용표 적용)
    slm_tokenizer : SLM HF tokenizer (동일 토크나이저로 카운트)
    prompt_share_mode: 'equal' or 'proportional'
    """
    N = min(3, len(base_parsed))
    step_texts = [ (base_parsed[i].get("chunk") or "") for i in range(N) ]

    # 각 스텝 출력 텍스트의 토큰 수(완성 토큰 근사)
    step_comp = [ _count_tokens_tok(slm_tokenizer, txt) for txt in step_texts ]
    tot_prompt = int(base_cost.get("prompt_tokens", 0))
    tot_comp   = int(base_cost.get("completion_tokens", 0))

    # 체인 완성 토큰과 스텝 합의 차이(예: 'FINAL:' 라인 등)를 마지막 스텝에 얹음
    rem = max(0, tot_comp - sum(step_comp))
    if rem > 0 and N > 0:
        step_comp[-1] += rem

    # 프롬프트 토큰 분배: 기본 equal, 옵션 proportional
    if N <= 0:
        return []
    if prompt_share_mode == "proportional" and sum(step_comp) > 0:
        wsum = float(sum(step_comp))
        step_prompt = [ int(round(tot_prompt * (c / wsum))) for c in step_comp ]
        diff = tot_prompt - sum(step_prompt)
        step_prompt[-1] += diff  # 반올림 보정
    else:
        base, r = divmod(tot_prompt, N)
        step_prompt = [base]*N
        step_prompt[-1] += r

    # 스텝별 비용 dict 생성 (+ api_cost 적용)
    out = []
    for i in range(N):
        c = {
            "prompt_tokens": step_prompt[i],
            "completion_tokens": step_comp[i],
            "latency_ms": 0.0
        }
        apply_api_cost(c, slm_model_name)  # MODEL_COST_CONFIG를 이용해 api_cost 산출
        out.append(c)
    return out


def parse_act_bias(raw: str, device):
    """
    CSV 'c,d,e' 형태를 받아 (Continue, Detract, Escalate) 순서의 텐서로 변환.
    예: --act_bias "-0.2,0.0,+0.3"
    """
    if not raw:
        return None
    try:
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != 3:
            return None
        return torch.tensor(parts, device=device, dtype=torch.float32)
    except Exception:
        return None



# ---- HF caller ----
class HFChatCaller:
    def __init__(self, model_name: str, dtype: Optional[str]="float16", seed: Optional[int]=None, hf_token: Optional[str]=None, profile: str="default"):
        self.model_name, self.profile, self.seed = model_name, (profile or "default").lower(), seed
        if torch.cuda.is_available():
            dt = {"bfloat16":torch.bfloat16, "bf16":torch.bfloat16, "float32":torch.float32, "fp32":torch.float32}.get(str(dtype).lower(), torch.float16)
        else: dt = torch.float32
        tok = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        auth = ({'token': tok, 'use_auth_token': tok} if tok else {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **auth)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **auth)
        fm_kwargs = dict(trust_remote_code=True, torch_dtype=dt, device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager", **auth)
        self.model = (AutoModelForSeq2SeqLM if getattr(cfg,"is_encoder_decoder",False) else AutoModelForCausalLM).from_pretrained(model_name, **fm_kwargs)
        self.model.eval()
    def _apply_template(self, messages: List[Dict[str,str]]) -> str:
        if isinstance(messages, str): messages = [{"role":"user","content":messages}]
        try: return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            lines=[f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages]; lines.append("ASSISTANT:"); return "\n".join(lines)
    @staticmethod
    def _looks_degenerate(text: str) -> bool:
        if not text: return True
        L=len(text); alnum=sum(c.isalnum() for c in text); nonascii=sum(ord(c)>127 for c in text)
        return (L>=80 and ((alnum/L)<0.25 or (nonascii/L)>0.50)) or ("!!!!!" in text and text.count("!!!!!")>=3)
    def _gen_once(
        self, inputs, eos_id, pad_id,
        *,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens: int = 512,
    ):
        try:
            gcfg = self.model.generation_config
            gcfg.do_sample = False
            gcfg.temperature = None
            gcfg.top_p = None
            gcfg.top_k = None
        except:
            pass

        lp = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
        kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            logits_processor=lp,
            renormalize_logits=True,
            do_sample=do_sample,
            temperature=(float(temperature) if do_sample and temperature else None),
            top_p=(float(top_p) if do_sample and top_p else None),
            top_k=(int(top_k) if do_sample and top_k is not None else None),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )
        if self.profile == "slm":
            # SLM은 항상 그리디
            kwargs.update(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        return out, (time.perf_counter() - t0) * 1000.0

    def chat(
        self,
        messages: List[Dict[str,str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        sample: Optional[bool] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> Tuple[str, Dict[str, float]]:
        prompt = self._apply_template(messages)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        enc = self.tokenizer(prompt, return_tensors="pt")
        dev = next(self.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        eos_id = getattr(self.model.config, "eos_token_id", None) or self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # 샘플링 여부 결정: 명시되면 그대로, 아니면 temperature>0 이면 켬
        use_sample = sample if sample is not None else (float(temperature) > 0.0 and self.profile != "slm")

        out1, lat1 = self._gen_once(
            enc, eos_id, pad_id,
            do_sample=use_sample,
            temperature=temperature if use_sample else None,
            top_p=top_p if use_sample else None,
            top_k=top_k if use_sample else None,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
        seq = out1.sequences[0]
        plen = int(enc["input_ids"].shape[-1])
        gen_ids = seq if getattr(self.model.config,"is_encoder_decoder",False) else seq[plen:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        latency_ms = lat1

        # (선택) 보정 샘플링은 이제 굳이 필요 없음. 원하면 유지:
        if (not use_sample) and self.profile != "slm" and self._looks_degenerate(text):
            out2, lat2 = self._gen_once(
                enc, eos_id, pad_id,
                do_sample=True, temperature=0.7, top_p=0.9, top_k=40,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
                max_new_tokens=min(max_new_tokens, 256),
            )
            seq2 = out2.sequences[0]
            gen_ids2 = seq2 if getattr(self.model.config,"is_encoder_decoder",False) else seq2[plen:]
            text2 = self.tokenizer.decode(gen_ids2, skip_special_tokens=True).strip()
            def _score(s):
                L = max(1, len(s)); alnum = sum(c.isalnum() for c in s)
                return (alnum / L, L)
            if _score(text2) > _score(text):
                text, gen_ids = text2, gen_ids2
            latency_ms += lat2

        cost = {
            "prompt_tokens": int(enc["input_ids"].shape[-1]),
            "completion_tokens": int(gen_ids.numel()),
            "latency_ms": float(latency_ms),
        }
        apply_api_cost(cost, self.model_name)
        return text, cost

    def close(self):
        try: del self.model; del self.tokenizer
        except: pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# ---- prompts ----
def prompt_decompose_compact(problem:str, qtype:str, n_steps:int=3)->List[Dict[str,str]]:
    if is_csqa():
        fewshot = """Example plan
Question:
Where do adults use glue sticks?
Options:
A. classroom
B. desk drawer
C. at school
D. office
E. kitchen drawer

To solve "xxx", do:
1. Identify the target entity/place.
2. Match the most typical option.
3. Decide the single best option."""
        user = f"""I will give you a multiple-choice question.
Decompose the reasoning into EXACTLY {n_steps} concise decision steps (≤10 words each).
Do not answer.

{fewshot}

Question:
{problem}

Answer Format (STRICT):
To solve "xxx", do:
1. ...
2. ...
3. ..."""
        return [
            {"role":"system","content":"Break the MCQA question into exactly 3 short decision steps. Do not answer."},
            {"role":"user","content":user},
        ]
    else:
        # GSM8K 그대로
        few = "Examples:\nTo solve 'A then B', do:\n1. Compute A.\n2. Compute B.\n3. Combine for final."
        user = f"""I will give you a math problem (type: {qtype}). Decompose it into EXACTLY 3 concise, numeric sub-questions (≤10 words each).
Rules: every given number/percent/price/unit must appear at least once.
{few}

Question:
{problem}

Answer Format (STRICT):
To solve "xxx", do:
1. ...
2. ...
3. ..."""
        return [
            {"role":"system","content":"Break the problem into exactly 3 atomic numeric steps."},
            {"role":"user","content":user},
        ]



def prompt_chain_raw(problem:str, steps:List[str], *, require_step_ans: bool=False)->List[Dict[str,str]]:
    s = "\n".join([f"Step {i+1}: {steps[i]}" for i in range(len(steps))])
    if is_csqa():
        fewshot = """Example
Question:
Where do adults use glue sticks?
Options:
A. classroom
B. desk drawer
C. at school
D. office
E. kitchen drawer

Steps to follow:
Step 1: Identify adult workplace.
Step 2: Match where glue sticks are used.
Step 3: Choose the best adult location.

Solution:
Step 1: Adults usually work in office. ANS: D
Step 2: Glue sticks are common office supplies. ANS: D
Step 3: Best adult location is office. ANS: D

FINAL: D
"""
        user = f"""Solve the multiple-choice question by following the listed steps and then give ONLY the letter.

Question:
{problem}

Steps to follow:
{s}

{fewshot}

Rules (STRICT):
- For EACH step, write ONE short line ending with: ANS:  <letter>.
- After Step 3, output exactly one line: FINAL: <letter>.
- No extra commentary after FINAL."""
        return [
            {"role":"system","content":"You solve MCQA step-by-step and end with 'FINAL: <letter>'."},
            {"role":"user","content":user},
        ]
    else:
        # GSM8K 그대로
        if require_step_ans:
            fewshot = """Example
Problem: Ducks lay 16/day. Eat 3. Bake 4. Sell rest @ $2. How many dollars?
Steps to follow:
Step 1: Compute eggs eaten.
Step 2: Compute eggs baked.
Step 3: Compute dollars from remaining eggs.

Solution:
Step 1: She eats 3 eggs. ANS: 3
Step 2: She bakes 4 eggs. ANS: 4
Step 3: Remaining = 16 - 3 - 4 = 9; dollars = 9 × 2 = 18. ANS: 18

FINAL: 18
"""
            user = f"""Solve the problem by following the listed steps and then give the final numeric answer.

Problem:
{problem}

Steps to follow:
{s}

{fewshot}

Rules (STRICT):
- For EACH step, write ONE short line ending with: ANS: <number>  (digits only).
- After Step 3, output exactly one line: FINAL: <number>  (digits only).
- No extra commentary after FINAL.

Begin."""
            return [
                {"role":"system","content":"You solve math step-by-step. Each step ends with 'ANS: <number>'. Finish with 'FINAL:'."},
                {"role":"user","content":user},
            ]
        else:
            user = f"""Solve the problem by following the listed steps and then give the final numeric answer.

Problem:
{problem}

Steps to follow:
{s}

Rules:
- For each step, write a small paragraph with the calculation/result.
- End the whole solution with a line 'FINAL: <number>' (digits only in the value).

Begin."""
            return [
                {"role":"system","content":"You solve math step-by-step and end with 'FINAL:'."},
                {"role":"user","content":user},
            ]


def prompt_llm_only(problem:str)->List[Dict[str,str]]:
    if is_csqa():
        fewshot = """Example
Question:
Where do adults use glue sticks?
Options:
A. classroom
B. desk drawer
C. at school
D. office
E. kitchen drawer

FINAL: D
"""
        user=f"""You will be given a multiple-choice question with options (options labeled A, B, C, ...).
Think briefly and return EXACTLY one line:

FINAL: <letter>

{fewshot}
Question:
{problem}"""
        return [
            {"role":"system","content":"Answer MCQA with a single letter. Output only 'FINAL: <letter>'."},
            {"role":"user","content":user},
        ]
    else:
        user=f"""Solve the following math problem carefully. Show brief reasoning and END WITH EXACTLY ONE LINE:
FINAL: <number>  (digits only)

Problem:
{problem}"""
        return [
            {"role":"system","content":"You are a careful math solver."},
            {"role":"user","content":user},
        ]


def prompt_finalize_attach_direct(problem: str, steps: List[str], executed_chunks: List[str]) -> List[Dict[str, str]]:
    lines = []
    for i, ch in enumerate(executed_chunks, 1):
        m = re.search(r'\bANS(?:WER)?\s*[:=]\s*([^\n]+)', (ch or ""), flags=re.I)
        ans = (m.group(1).strip() if m else "UNSURE")
        lines.append(f"- Step {i}: {steps[i-1]} -> ANS: {ans}")

    if is_csqa():
        user = (
            "Choose ONLY the final option letter (A-E) by relying on the quoted keywords and reasons in the notes.\n\n"
            f"Question (with options):\n{problem}\n\n"
            "Notes from executed steps (may be imperfect):\n" + "\n".join(lines) + "\n\n"
            "END WITH EXACTLY ONE LINE:\nFINAL: <letter>   # letter only"
        )
        return [
            {"role": "system", "content": "Finalize MCQA using the given step notes and their quoted keywords. Output only the letter."},
            {"role": "user", "content": user},
        ]
    else:
        user = (
            "Solve the problem carefully and provide ONLY the final numeric answer.\n\n"
            f"Problem:\n{problem}\n\n"
            "Helpful notes (may be incomplete or wrong):\n" + "\n".join(lines) + "\n\n"
            "END WITH EXACTLY ONE LINE:\nFINAL: <number>"
        )
        return [
            {"role": "system", "content": "You are a careful math solver."},
            {"role": "user", "content": user},
        ]


def prompt_finalize_from_steps_strict(problem: str, steps: List[str], executed_chunks: List[str]) -> List[Dict[str, str]]:
    lines=[]
    for i,ch in enumerate(executed_chunks,1):
        m=re.search(r'\bANS(?:WER)?\s*[:=]\s*([^\n]+)', (ch or ""), flags=re.I)
        lines.append(f"- Step {i}: {steps[i-1]} -> {'ANS: '+m.group(1).strip() if m else 'UNSURE'}")

    if is_csqa():
        user=(
            "Use ONLY the provided sub-results (and their quoted keywords). Do NOT recompute earlier steps.\n\n"
            f"Context:\n" + "\n".join(lines) + f"\n\nOriginal question (with options):\n{problem}\n\n"
            "Output EXACTLY ONE LINE:\nFINAL: <letter>   # letter only"
        )
        return [
            {"role":"system","content":"Compute the final MCQA letter strictly from provided sub-results. End with 'FINAL:'."},
            {"role":"user","content":user},
        ]
    else:
        user=(f"Use ONLY the provided sub-results below to compute the final answer. Do NOT recompute earlier steps.\n\n"
              f"Context:\n" + "\n".join(lines) + f"\n\nOriginal problem:\n{problem}\n\n"
              "Output EXACTLY ONE LINE:\nFINAL: <number>")
        return [
            {"role":"system","content":"Compute the final result strictly from provided sub-results. End with 'FINAL:'."},
            {"role":"user","content":user},
        ]


def prompt_force_final_only(problem:str, prev_text:str)->List[Dict[str,str]]:
    if is_csqa():
        user=f"""You wrote the following reasoning for the question below.

Question:
{problem}

Your text:
\"\"\"{(prev_text or '').strip()[:1200]}\"\"\"

Now output EXACTLY ONE LINE, NOTHING ELSE:
FINAL: <letter>"""
        return [{"role":"user","content":user}]
    else:
        user=f"""You wrote the following reasoning for the problem below.

Problem:
{problem}

Your text:
\"\"\"{(prev_text or '').strip()[:1200]}\"\"\"

Now output EXACTLY ONE LINE, NOTHING ELSE:
FINAL: <number>  (digits only)"""
        return [{"role":"user","content":user}]

def prompt_edit_raw_chunk_minimal(problem:str, subproblem:str, current_chunk:str, supports: List[Tuple[str,str]])->List[Dict[str,str]]:
    support_block = ""
    if supports:
        support_block = "<EXAMPLES>\n" + "\n\n".join([f"[{i}] Subproblem: {sq}\nResponse:\n{resp}" for i,(sq,resp) in enumerate(supports,1)]) + "\n</EXAMPLES>\n"

    if is_csqa():
        user = f"""You are editing ONE step of a 5-option MCQA (A-E).

Return EXACTLY TWO lines:
Line 1: <= 12 words giving the best option's reason; quote 1-3 keywords from that option in "double quotes". Do NOT include the option letter here.
Line 2: ANS: <letter>    

Original question (with options):
{problem}

Subproblem (this step only):
{subproblem}

Current step text:
\"\"\"{(current_chunk or '').strip()[:600]}\"\"\"

Constraints (STRICT):
- Do NOT restate the full question or list options again.
- Do NOT output anything after Line 2.
- Keep the scope to THIS subproblem only.
{support_block}Return only the two lines specified."""
        return [{"role":"user","content":user}]
    else:
        user=f"""EDIT minimally and return EXACTLY TWO lines:
Line 1: one short sentence solving THIS subproblem only.
Line 2: ANS: <number>   # digits only

Original problem:
{problem}

Subproblem:
{subproblem}

Current step text:
\"\"\"{(current_chunk or '').strip()[:600]}\"\"\"

{support_block}Return only the two lines specified."""
        return [{"role":"user","content":user}]



def prompt_llm_direct_with_internal_steps(
    problem: str,
    n_steps: int = 3,
    require_step_ans: bool = True
) -> List[Dict[str, str]]:
    if is_csqa():
        rules = (
            "Rules (STRICT):\n"
            f"- Write \"Steps to follow:\" with exactly {n_steps} steps labeled 'Step 1:' ... 'Step {n_steps}:'.\n"
            "- In the \"Solution:\" section, for EACH step write ONE short line ending with: ANS: <letter>.\n"
            "- After the steps, output exactly one line: FINAL: <letter>.\n"
            "- No extra commentary after FINAL.\n"
        )
        user = (
            "Solve the multiple-choice question by following the listed steps and then provide the final letter.\n\n"
            f"Question:\n{problem}\n\n"
            f"First, write \"Steps to follow:\" with exactly {n_steps} concise steps. Then write \"Solution:\".\n\n"
            f"{rules}\n"
            "Begin."
        )
        return [
            {"role": "system", "content": "You solve MCQA with internal steps and end with 'FINAL: <letter>'."},
            {"role": "user", "content": user},
        ]

    # GSM8K branch (원래 내용 유지)
    fewshot = """Example
Problem: Ducks lay 16/day. Eat 3. Bake 4. Sell rest @ $2. How many dollars?
Steps to follow:
Step 1: Compute eggs eaten.
Step 2: Compute eggs baked.
Step 3: Compute dollars from remaining eggs.

Solution:
Step 1: She eats 3 eggs. ANS: 3
Step 2: She bakes 4 eggs. ANS: 4
Step 3: Remaining = 16 - 3 - 4 = 9; dollars = 9 × 2 = 18. ANS: 18

FINAL: 18
"""

    if require_step_ans:
        rules = (
            "Rules (STRICT):\n"
            f"- Write \"Steps to follow:\" with exactly {n_steps} steps labeled 'Step 1:' ... 'Step {n_steps}:'.\n"
            "- Every given number/percent/price/unit in the problem must appear at least once across the steps.\n"
            "- In the \"Solution:\" section, for EACH step write ONE short line ending with: ANS: <number>  (digits only).\n"
            f"- After Step {n_steps}, output exactly one line: FINAL: <number>  (digits only).\n"
            "- No extra commentary after FINAL.\n"
        )
    else:
        rules = (
            "Rules:\n"
            "- For each step, write a small paragraph with the calculation/result.\n"
            f"- End the whole solution with a line 'FINAL: <number>' (digits only in the value).\n"
        )

    user = (
        "Solve the problem by following the listed steps and then give the final numeric answer.\n\n"
        f"Problem:\n{problem}\n\n"
        f"First, write \"Steps to follow:\" with exactly {n_steps} concise steps labeled "
        f"'Step 1:' ... 'Step {n_steps}:'. Then write \"Solution:\" that follows those steps.\n\n"
        f"{fewshot}\n"
        f"{rules}\n"
        "Begin."
    )

    return [
        {"role": "system", "content": "You solve math step-by-step. Each step ends with 'ANS: <number>'. Finish with 'FINAL:'."},
        {"role": "user", "content": user},
    ]

# ---- raw-chain parsing ----
STEP_SPLIT_RE = re.compile(r'(?:^|\n)\s*(?:\*\*?\s*)?(?:step\s*[1-3]\b|[1-3]\s*[:.)-])', flags=re.I)
def split_raw_chain_into_chunks(text:str, n:int=3)->List[str]:
    if not text: return [""]*n
    idxs=[m.start() for m in STEP_SPLIT_RE.finditer(text)]
    mfin = re.search(r'^\s*(?:final(?:\s*answer)?|answer|result)\s*[:=]\s*.+$', text, flags=re.I|re.M)
    endpos = mfin.start() if mfin else len(text)
    chunks=[]
    if len(idxs)>=n:
        idxs=idxs[:n]+[endpos]
        for a,b in zip(idxs, idxs[1:]): chunks.append(text[a:b].strip())
    else:
        L=endpos; cut=[int(L*i/n) for i in range(n)]+[L]
        for i in range(n): chunks.append(text[cut[i]:cut[i+1]].strip())
    return (chunks+[""]*n)[:n]

# ---- parse ANS in a chunk ----
KEY_NEAR = ["final","total","profit","revenue","dollar","value","increase","remaining","meter","meters","cup","cups","egg","eggs","cost","price"]
def extract_ans_from_chunk(chunk: str, *, strict: bool = False) -> Tuple[bool, bool, str]:
    """
    Returns: has_ans, is_valid, value
    - GSM8K: value는 숫자, is_valid=True iff 숫자
    - CSQA : value는 [A-E], is_valid=True iff A-E
    """
    if not chunk: return False, False, ""
    text=chunk.strip()
    # 1) 우선 ANS: 패턴
    m = re.search(r'\bANS(?:WER)?\s*[:=]\s*([^\n]+)', text, flags=re.I)
    if m:
        val_raw=m.group(1).strip()
        # GSM8K: 숫자
        num=last_number(val_raw)
        if not is_csqa() and num:
            return True, True, num
        # CSQA: A-E
        let=extract_choice_letter(val_raw)
        if is_csqa() and let:
            return True, True, let
        return True, False, val_raw
    if strict:
        return False, False, ""
    # 2) 관대 모드
    if is_csqa():
        let=extract_choice_letter(text)
        if let: return True, True, let
    m2 = re.search(r'=\s*[\$]?\s*(-?\d[\d,]*(?:\.\d+)?)\b', text)
    if m2: return True, True, m2.group(1).replace(",","")
    cands=[]
    for m3 in re.finditer(NUM_WITH_COMMAS, text):
        raw=m3.group(0); norm=raw.replace(",",""); pos=m3.start()
        win=text[max(0,pos-24): pos+24].lower(); near_kw=any(k in win for k in KEY_NEAR)
        score=(1 if near_kw else 0, -pos); cands.append((score,norm))
    if cands:
        cands.sort(key=lambda x:x[0], reverse=True); return True, True, cands[0][1]
    return False, False, ""

# ---- SLM signal (router input) ----
def slm_sig_from_text(text: str) -> np.ndarray:
    s=(text or ""); chars=list(s); L=max(1,len(chars))
    return np.array([
        1.0 if s.strip() else 0.0,
        1.0 if re.search(r'-?\d+(?:\.\d+)?', s) else 0.0,
        sum(c.isdigit() for c in chars)/L,
        sum(c in "+-*/=^" for c in chars)/L,
        float(len(s.split()))
    ], dtype=np.float32)

# ---- retrieval (tiny BM25) ----
_WORD_RE = re.compile(r'[가-힣]+|[a-zA-Z0-9]+')
def _tok(text:str)->List[str]: return _WORD_RE.findall((text or "").lower())
class SimpleBM25:
    def __init__(self, docs:List[List[str]], k1:float=1.5, b:float=0.75):
        self.docs=docs; self.N=len(docs); self.k1=k1; self.b=b
        self.doc_len=[len(d) for d in docs]; self.avgdl=(sum(self.doc_len)/self.N) if self.N>0 else 0.0
        self.df={}
        for d in docs:
            for w in set(d): self.df[w]=self.df.get(w,0)+1
        self.idf={w: math.log((self.N - df + 0.5)/(df + 0.5) + 1.0) for w,df in self.df.items()}
    def score(self, q_tokens:List[str], idx:int)->float:
        d=self.docs[idx]; dl=self.doc_len[idx] if idx < len(self.doc_len) else 0
        if dl==0 or self.avgdl==0: return 0.0
        tf={}
        for w in d: tf[w]=tf.get(w,0)+1
        s=0.0; q_tokens=_tok(" ".join(q_tokens)) if isinstance(q_tokens, list) else _tok(str(q_tokens))
        for w in q_tokens:
            if w not in tf or w not in self.idf: continue
            idf=self.idf[w]; f=tf[w]; denom=f + self.k1*(1 - self.b + self.b*dl/self.avgdl)
            s += idf * (f*(self.k1+1)) / (denom if denom>0 else 1e-9)
        return s
    def topk(self, q_text:str, k:int)->List[int]:
        q=_tok(q_text); scores=[(i,self.score(q,i)) for i in range(self.N)]; scores.sort(key=lambda x:x[1], reverse=True)
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
                    if "subq" in obj and "resp" in obj: bank.append(obj)
                except: continue
    except FileNotFoundError: pass
    return bank

def build_bm25_from_bank(bank:List[Dict[str,Any]])->SimpleBM25:
    docs=[_tok(str(rec.get("subq",""))) for rec in bank]
    return SimpleBM25(docs)

def retrieve_support_pairs_bm25(bank: List[Dict[str,Any]], bm25: Optional[SimpleBM25], query_subq: str, topk: int, maxchars: int) -> List[Tuple[str,str]]:
    if not bank or topk<=0: return []
    idxs=bm25.topk(query_subq, topk) if bm25 else list(range(min(topk, len(bank))))
    pairs=[]
    for k in idxs:
        rec=bank[k]; sq=str(rec.get("subq","")).strip(); resp=str(rec.get("resp","")).strip()
        if maxchars and maxchars>0 and len(resp)>maxchars: resp=resp[:maxchars].rstrip()
        pairs.append((sq, resp))
    return pairs

# ---- Router model (inference only) ----
class MeanPooler(nn.Module):
    def forward(self, last_hidden, attn_mask): m=attn_mask.unsqueeze(-1).float(); return (last_hidden*m).sum(1)/m.sum(1).clamp(min=1e-6)
class BertEncoder(nn.Module):
    def __init__(self, name="bert-base-uncased", max_len=384, trainable=True, local_files_only=False):
        super().__init__(); self.max_len=max_len; self.pool=MeanPooler()
        self.tok=AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)
        self.bert=AutoModel.from_pretrained(name, local_files_only=local_files_only)
        if not trainable:
            for p in self.bert.parameters(): p.requires_grad=False
    def encode_texts(self, texts: List[str], device):
        enc=self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc={k:v.to(device) for k,v in enc.items()}
        with torch.no_grad(): out=self.bert(**enc); pooled=self.pool(out.last_hidden_state, enc["attention_mask"])
        return pooled
class GRURouter(nn.Module):
    def __init__(self, num_models, aux_dim, slm_sig_dim, bert_name="bert-base-uncased", max_len=384, freeze_bert=True, gru_hidden=384, step_emb_dim=16, proj_dim=512, p_drop=0.1, bert_local_files_only=False):
        super().__init__()
        self.enc=BertEncoder(bert_name, max_len=max_len, trainable=not freeze_bert, local_files_only=bert_local_files_only)
        self.gru=nn.GRU(768, gru_hidden, batch_first=True); self.step_emb=nn.Embedding(64, step_emb_dim)
        self.model_vecs=nn.Parameter(torch.randn(num_models,768)*0.02); self.sim_temp=nn.Parameter(torch.tensor(0.7))
        self._aux_dim=int(aux_dim); self._slm_dim=int(slm_sig_dim)
        cur_in=768+768+3+self._aux_dim+self._slm_dim+step_emb_dim
        self.cur_mlp=nn.Sequential(nn.Linear(cur_in, proj_dim), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(proj_dim, proj_dim), nn.ReLU())
        self.fuse=nn.Sequential(nn.Linear(gru_hidden+proj_dim, proj_dim), nn.ReLU())
        self.dropout=nn.Dropout(p_drop); self.ln_fuse=nn.LayerNorm(proj_dim)
        self.act_head=nn.Linear(proj_dim,3); self.mdl_head=nn.Linear(proj_dim,num_models)
        self.register_buffer("aux_mean", torch.zeros(self._aux_dim)); self.register_buffer("aux_std", torch.ones(self._aux_dim))
        self.register_buffer("slm_mean", torch.zeros(self._slm_dim)); self.register_buffer("slm_std", torch.ones(self._slm_dim))
    def set_feature_stats(self, aux_mean, aux_std, slm_mean, slm_std, device):
        def to_t(x,d,fill):
            if isinstance(x,(list,tuple)):
                return torch.tensor(x, device=device, dtype=torch.float32) if len(x)>0 else torch.full((d,), fill, device=device, dtype=torch.float32)
            if isinstance(x,np.ndarray): return torch.from_numpy(x).to(device=device, dtype=torch.float32)
            if isinstance(x,torch.Tensor): return x.to(device=device, dtype=torch.float32)
            return torch.full((d,), fill, device=device, dtype=torch.float32)
        aux_mean,aux_std,slm_mean,slm_std = to_t(aux_mean,self._aux_dim,0.0), to_t(aux_std,self._aux_dim,1.0).clamp(min=1e-6), to_t(slm_mean,self._slm_dim,0.0), to_t(slm_std,self._slm_dim,1.0).clamp(min=1e-6)
        with torch.no_grad():
            if aux_mean.numel()==self._aux_dim: self.aux_mean.copy_(aux_mean)
            if aux_std.numel()==self._aux_dim: self.aux_std.copy_(aux_std)
            if slm_mean.numel()==self._slm_dim: self.slm_mean.copy_(slm_mean)
            if slm_std.numel()==self._slm_dim: self.slm_std.copy_(slm_std)
    def encode_units(self, batch_units: List[List[str]], device):
        flat=[]; offsets=[0]
        for units in batch_units: flat+=units; offsets.append(offsets[-1]+len(units))
        Z=self.enc.encode_texts(flat, device=device); lengths=[]; chunks=[]
        for i in range(len(batch_units)):
            a,b=offsets[i],offsets[i+1]; chunks.append(Z[a:b]); lengths.append(b-a)
        Tmax=max(lengths) if lengths else 1; B=len(batch_units); X=torch.zeros(B,Tmax,768,device=Z.device)
        for i,ch in enumerate(chunks):
            if ch.numel()>0: X[i,:ch.size(0),:]=ch
        return X,lengths
    def forward(self, batch_units: List[List[str]], q_texts: List[str], slm_sigs_in: torch.Tensor, step_idx: torch.Tensor):
        self.gru.flatten_parameters(); device=step_idx.device
        X,lengths=self.encode_units(batch_units, device=device)
        packed=torch.nn.utils.rnn.pack_padded_sequence(X, lengths=lengths, batch_first=True, enforce_sorted=False)
        _,h=self.gru(packed); h_last=h.squeeze(0)
        B=X.size(0); q_emb=torch.stack([X[i,max(0,lengths[i]-1),:] for i in range(B)], dim=0)
        past_mean=[]
        for i in range(B):
            L=max(0,lengths[i]-1); past_mean.append(torch.zeros(768,device=device) if L<=0 else X[i,:L,:].mean(0))
        past_mean=torch.stack(past_mean,dim=0)
        sims=torch.matmul(q_emb, self.model_vecs.t()); prob=F.softmax(self.sim_temp*sims, dim=1).clamp(min=1e-6)
        sim_max=sims.max(1,keepdim=True).values; sim_min=sims.min(1,keepdim=True).values; sim_ent=-(prob*prob.log()).sum(1,keepdim=True)
        def aux_vec_from_qtext(q_text: str) -> np.ndarray:
            s=(q_text or "").replace("[Q]","").strip().lower(); tokens=s.split(); T=max(1,len(tokens)); chars=list(s); L=max(1,len(chars))
            numbers_ratio=float(len(re.findall(r'\b\d+(\.\d+)?\b', s)))/T; units_flag=1.0 if re.search(r"\b(cm|mm|km|kg|g|miles?|hours?|mins?|seconds?|percent|%)\b", s) else 0.0
            ops_ratio=sum(c in "+-*/=^" for c in chars)/L; avg_toklen=(np.mean([len(t) for t in tokens]) if tokens else 0.0)
            has_eq_kw=1.0 if any(k in s for k in ["equation","system","integral","derivative"]) else 0.0
            has_reason_kw=1.0 if any(k in s for k in ["prove","show","explain","why"]) else 0.0
            return np.array([numbers_ratio, units_flag, ops_ratio, avg_toklen, has_eq_kw, has_reason_kw], dtype=np.float32)
        AUX=np.stack([aux_vec_from_qtext(t) for t in q_texts], axis=0); aux=torch.from_numpy(AUX).to(device).float()
        # ★ NEW: aux 차원을 모델 기대치에 맞춤
        if self._aux_dim == 0:
            aux = aux.new_zeros((aux.size(0), 0))
        elif aux.size(1) > self._aux_dim:
            aux = aux[:, :self._aux_dim]
        elif aux.size(1) < self._aux_dim:
            pad = aux.new_zeros((aux.size(0), self._aux_dim - aux.size(1)))
            aux = torch.cat([aux, pad], dim=1)
        aux=(aux-self.aux_mean)/self.aux_std.clamp(min=1e-6); slm_sigs=(slm_sigs_in.float()-self.slm_mean)/self.slm_std.clamp(min=1e-6)
        stepv=self.step_emb(step_idx.clamp(min=0, max=self.step_emb.num_embeddings-1))
        cur_feat=torch.cat([q_emb,past_mean,sim_max,sim_min,sim_ent,aux,slm_sigs,stepv],dim=1)
        cur_h=self.cur_mlp(cur_feat)
        fused=self.fuse(torch.cat([h_last,cur_h],dim=1)); fused=self.ln_fuse(self.dropout(fused))
        return self.act_head(fused), self.mdl_head(fused)

def forbid_slm_on_de(mdl_logits: torch.Tensor, act_ids: torch.Tensor, slm_idx: int = 0):
    out=mdl_logits.clone()
    if out.numel()==0: return out
    mask=(act_ids!=0)
    if not mask.any(): return out
    out[mask, slm_idx]=out.new_tensor(-1e9)
    return out

# def load_router(ckpt_path: str, device):
#     ckpt=torch.load(ckpt_path, map_location=device)
#     model_names=ckpt.get("model_names") or ckpt.get("allowed_models") or ["SLM","Qwen/Qwen2.5-7B-Instruct","meta-llama/Llama-3.1-8B-Instruct","Qwen/Qwen2.5-14B-Instruct"]
#     cfg=ckpt.get("config") or {}; bert_name=ckpt.get("bert") or cfg.get("bert_name","bert-base-uncased")
#     max_len=ckpt.get("max_len", cfg.get("max_len",384)); freeze_bert=bool(ckpt.get("freeze_bert", cfg.get("freeze_bert",True)))
#     fs=ckpt.get("feature_stats", {}); aux_dim=len(fs.get("aux_mean",[])) or cfg.get("aux_dim",6); slm_dim=len(fs.get("slm_mean",[])) or cfg.get("slm_sig_dim",5)
#     model=GRURouter(num_models=len(model_names), aux_dim=aux_dim, slm_sig_dim=slm_dim, bert_name=bert_name, max_len=max_len, freeze_bert=freeze_bert)
#     sd=ckpt.get("state_dict") or ckpt; model.load_state_dict(sd, strict=False)
#     model.set_feature_stats(fs.get("aux_mean",[0.0]*aux_dim), fs.get("aux_std",[1.0]*aux_dim), fs.get("slm_mean",[0.0]*slm_dim), fs.get("slm_std",[1.0]*slm_dim), device=device)
#     la_bias=ckpt.get("la_bias", None)
#     if la_bias is not None: la_bias=torch.as_tensor(la_bias, device=device).float()
#     model.to(device).eval()
#     return model, model_names, la_bias

def load_router(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    model_names = ckpt.get("model_names") or ckpt.get("allowed_models") or [
        "SLM","Qwen/Qwen2.5-7B-Instruct","meta-llama/Llama-3.1-8B-Instruct","Qwen/Qwen2.5-14B-Instruct"
    ]
    cfg = ckpt.get("config") or {}
    bert_name   = ckpt.get("bert") or cfg.get("bert_name","bert-base-uncased")
    max_len     = ckpt.get("max_len", cfg.get("max_len",384))
    freeze_bert = bool(ckpt.get("freeze_bert", cfg.get("freeze_bert",True)))

    fs = ckpt.get("feature_stats", {})

    # ❌ 기존: aux_dim = len(fs.get("aux_mean", [])) or cfg.get("aux_dim", 6)
    # ✅ 수정: 통계가 "있으면" 그 길이를 그대로 쓰고(0 포함), 없으면 cfg/디폴트
    if "aux_mean" in fs:
        aux_dim = len(fs["aux_mean"])           # 0도 유효
    else:
        aux_dim = int(cfg.get("aux_dim", 6))

    if "slm_mean" in fs:
        slm_dim = len(fs["slm_mean"])           # 0도 유효 가능
    else:
        slm_dim = int(cfg.get("slm_sig_dim", 5))

    model = GRURouter(
        num_models=len(model_names),
        aux_dim=aux_dim,
        slm_sig_dim=slm_dim,
        bert_name=bert_name,
        max_len=max_len,
        freeze_bert=freeze_bert
    )

    sd = ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)

    model.set_feature_stats(
        fs.get("aux_mean", [0.0]*aux_dim),
        fs.get("aux_std",  [1.0]*aux_dim),
        fs.get("slm_mean", [0.0]*slm_dim),
        fs.get("slm_std",  [1.0]*slm_dim),
        device=device
    )

    la_bias = ckpt.get("la_bias", None)
    if la_bias is not None:
        la_bias = torch.as_tensor(la_bias, device=device).float()

    ablate_flags = set((ckpt.get("ablate") or []))

    model.to(device).eval()
    print(f"[INFO] Router feature dims: aux_dim={aux_dim}, slm_dim={slm_dim}, ablate={sorted(list(ablate_flags))}")
    return model, model_names, la_bias, ablate_flags


# def predict_action_and_model(router: GRURouter, model_names: List[str], units: List[str], q_text: str,
#                              slm_sig_vec: np.ndarray, step_idx: int,
#                              la_bias: Optional[torch.Tensor], la_bias_disabled: bool,
#                              device, act_bias_delta: Optional[torch.Tensor] = None):
#     with torch.no_grad():
#         act_logits, mdl_logits = router([units], [q_text],
#                                         torch.from_numpy(slm_sig_vec).unsqueeze(0).to(device),
#                                         torch.tensor([step_idx], dtype=torch.long, device=device))
#         # 1) 체크포인트 내장 la_bias (옵션)
#         if (la_bias is not None) and (not la_bias_disabled):
#             act_logits = act_logits + la_bias.view(1, -1)
#         # 2) CLI로 주는 추가 바이어스(항상 적용: 실험 제어용)
#         if act_bias_delta is not None:
#             act_logits = act_logits + act_bias_delta.view(1, -1)

#         pa = int(act_logits.argmax(1).item())
#         blended = forbid_slm_on_de(mdl_logits, torch.tensor([pa], device=device), slm_idx=0)
#         pm = int(blended.argmax(1).item())
#         if pa != 0 and pm == 0 and blended.size(1) > 1:
#             vals = blended[0].detach().cpu().numpy()
#             order = np.argsort(-vals)
#             for idx in order:
#                 if idx != 0:
#                     pm = int(idx); break
#         return ACTIONS[pa], model_names[pm]

def _fit_vec_to_dim(t: torch.Tensor, target_dim: int) -> torch.Tensor:
    # t: (B, D_in) 또는 (D_in,) -> (B, target_dim)
    if t.dim() == 1: t = t.unsqueeze(0)
    B, Din = t.size(0), t.size(1)
    if target_dim == Din:
        return t
    if target_dim == 0:
        return t.new_zeros((B, 0))
    if Din == 0:
        return t.new_zeros((B, target_dim))
    if Din > target_dim:
        return t[:, :target_dim]
    # Din < target_dim : zero-pad
    pad = t.new_zeros((B, target_dim - Din))
    return torch.cat([t, pad], dim=1)

def predict_action_and_model(router: GRURouter, model_names: List[str], units: List[str], q_text: str,
                             slm_sig_vec: np.ndarray, step_idx: int,
                             la_bias: Optional[torch.Tensor], la_bias_disabled: bool,
                             device, act_bias_delta: Optional[torch.Tensor] = None):
    with torch.no_grad():
        # 기존 slm_sig_vec → router._slm_dim에 맞춤
        slm_sig_t = torch.from_numpy(slm_sig_vec).float().to(device).unsqueeze(0)
        # router 객체에 기대 차원 보관됨
        slm_sig_t = _fit_vec_to_dim(slm_sig_t, router._slm_dim)

        act_logits, mdl_logits = router(
            [units], [q_text], slm_sig_t, torch.tensor([step_idx], dtype=torch.long, device=device)
        )

        if act_bias_delta is not None:
            act_logits = act_logits + act_bias_delta.view(1, -1)

        # ★ NEW: ablation 반영
        # 1) la ablated면 la_bias 강제 미적용
        if (la_bias is not None) and (not la_bias_disabled):
            act_logits = act_logits + la_bias.view(1, -1)

        # 2) forbid ablated면 SLM 금지 규칙 적용하지 않음
        if getattr(router, "_ablate_forbid", False):
            blended = mdl_logits
        else:
            blended = forbid_slm_on_de(mdl_logits, torch.tensor([act_logits.argmax(1).item()], device=device), slm_idx=0)

        pa = int(act_logits.argmax(1).item())
        pm = int(blended.argmax(1).item())
        if pa != 0 and pm == 0 and blended.size(1) > 1:
            vals = blended[0].detach().cpu().numpy()
            order = np.argsort(-vals)
            for idx in order:
                if idx != 0:
                    pm = int(idx); break
        return ACTIONS[pa], model_names[pm]



# ---- runners ----
def run_detract_on_chunk(problem, subproblem, current_chunk, caller: HFChatCaller, supports: List[Tuple[str,str]]):
    msgs=prompt_edit_raw_chunk_minimal(problem, subproblem, current_chunk, supports)
    text, cost = caller.chat(msgs, max_new_tokens=160, temperature=0.0)
    lines=[ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    chunk_out = (lines[0]+"\n"+lines[1]) if len(lines)>=2 else text.strip()
    has,isnum,val=extract_ans_from_chunk(chunk_out, strict=True)
    if not has: has,isnum,val=extract_ans_from_chunk(current_chunk)
    if not isnum:
        num_or_letter = (extract_choice_letter(chunk_out) if is_csqa() else last_number(chunk_out))
        if num_or_letter:
            chunk_out=f"{chunk_out}\nANS: {num_or_letter}"
            has,isnum,val=True,True,num_or_letter
    return chunk_out, has, isnum, val, cost

# def run_escalate_to_final(problem, steps, executed_chunks, start_idx, caller: HFChatCaller):
#     msgs=prompt_llm_only(problem); text, cost = caller.chat(msgs, max_new_tokens=400, temperature=0.0)
#     final=extract_final_line(text) or last_number(text) or ""
#     if is_csqa() and not final:
#         # CSQA fallback: 그냥 본문에서 A-E 하나 집기
#         final = extract_choice_letter(text) or ""
#     if not final:
#         text2, cost2 = caller.chat(prompt_force_final_only(problem, text), max_new_tokens=16, temperature=0.0)
#         final2=extract_final_line(text2)
#         for k in cost: cost[k]+=cost2.get(k,0.0)
#         if final2: final=final2
#     return (final if final else None), cost

def run_escalate_to_final(problem, steps, executed_chunks, start_idx, caller: HFChatCaller):
    # 1) 본 에스컬레이트 호출: 샘플링 설정 복붙
    msgs = prompt_llm_only(problem)
    text, cost = caller.chat(
        msgs,
        max_new_tokens=400,
        sample=True,              # ← 샘플링 ON
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    )

    final = extract_final_line(text) or last_number(text) or ""
    if is_csqa() and not final:
        final = extract_choice_letter(text) or ""

    # 2) FINAL 강제 재시도는 그리디 유지(안전). 동일 옵션 쓰고 싶으면 아래 temperature=0.0을 위와 같이 바꾸면 됨.
    if not final:
        text2, cost2 = caller.chat(
            prompt_force_final_only(problem, text),
            max_new_tokens=16,
            temperature=0.0
        )
        final2 = extract_final_line(text2)
        if is_csqa() and not final2:
            final2 = extract_choice_letter(text2)
        for k in cost:
            cost[k] += cost2.get(k, 0.0)
        if final2:
            final = final2

    return (final if final else None), cost

def run_finalize(problem: str, steps: List[str], executed_chunks: List[str], caller: HFChatCaller, *, max_new_tokens: int = 80, mode: str = "attach_direct"):
    msgs = prompt_llm_only(problem) if mode=="direct" else (prompt_finalize_from_steps_strict(problem, steps, executed_chunks) if mode=="from_steps" else prompt_finalize_attach_direct(problem, steps, executed_chunks))
    text, cost = caller.chat(msgs, max_new_tokens=max_new_tokens, temperature=0.0)
    final = extract_final_line(text)
    if not final and is_csqa():
        final = extract_choice_letter(text)
    if not final:
        text2, cost2 = caller.chat(prompt_force_final_only(problem, text), max_new_tokens=16, temperature=0.0)
        f2=extract_final_line(text2)
        if not f2 and is_csqa(): f2 = extract_choice_letter(text2)
        if f2: final=f2
        for k in cost: cost[k]+=cost2.get(k,0.0)
    return (final or ""), cost, text

# ---- Router Finalize AUTO (from_steps vs direct, cross-check) ----
def _chunk_has_numeric_or_letter_ans(ch: str) -> bool:
    has,isvalid,val = extract_ans_from_chunk(ch, strict=True)
    if has and isvalid and str(val).strip(): return True
    has2,isvalid2,val2 = extract_ans_from_chunk(ch, strict=False)
    return bool(has2 and isvalid2 and str(val2).strip())

def steps_look_reliable(executed_chunks: list) -> bool:
    if not executed_chunks or len(executed_chunks)<3: return False
    ok=[]
    for ch in executed_chunks[:3]:
        if not ch or len(ch)>6000: ok.append(False); continue
        ok.append(_chunk_has_numeric_or_letter_ans(ch))
    return all(ok)

def run_finalize_router_auto(problem: str, steps: list, executed_chunks: list, caller: HFChatCaller, *, max_new_tokens: int = 80):
    mode = "attach_direct" if steps_look_reliable(executed_chunks) else "direct"
    fin, c, _ = run_finalize(problem, steps, executed_chunks, caller, max_new_tokens=max_new_tokens, mode=mode)
    return fin, c, mode

# ---- baselines ----
def run_singlepass_chain(slm:HFChatCaller, problem:str, steps:List[str], *, require_step_ans: bool=False):
    msgs = prompt_chain_raw(problem, steps, require_step_ans=require_step_ans)
    text, cost = slm.chat(msgs, max_new_tokens=1000, temperature=0.0)

    final = extract_final_line(text) or last_number(text) or ""
    if is_csqa() and not final:
        final = extract_choice_letter(text) or ""

    if not final:
        forced, cost2 = slm.chat(prompt_force_final_only(problem, text), max_new_tokens=16, temperature=0.0)
        f2 = extract_final_line(forced)
        if not f2 and is_csqa(): f2 = extract_choice_letter(forced)
        if f2:
            final = f2
        for k in cost:
            cost[k] += cost2.get(k, 0.0)

    chunks = split_raw_chain_into_chunks(text, n=3)
    parsed=[]
    for ch in chunks:
        has, isnum, val = extract_ans_from_chunk(ch)  # 관대 모드
        parsed.append({"chunk": ch, "has": has, "isnum": isnum, "value": val})

    return parsed, str(final), text, cost

def run_llm_only(caller: HFChatCaller, problem: str, steps: Optional[List[str]] = None, *, mode: str = "direct"):
    def _direct():
        msgs = prompt_llm_only(problem)
        text, cost = caller.chat(msgs, max_new_tokens=600, temperature=0.0)
        final = extract_final_line(text) or ""
        if is_csqa() and not final:
            final = extract_choice_letter(text) or ""
        return str(final), text, cost

    def _direct_steps(n_steps=3, require_step_ans=True):
        msgs = prompt_llm_direct_with_internal_steps(problem, n_steps=n_steps, require_step_ans=require_step_ans)
        text, cost = caller.chat(msgs, max_new_tokens=900, temperature=0.0)
        final = extract_final_line(text) or ""
        if is_csqa() and not final:
            final = extract_choice_letter(text) or ""
        return str(final), text, cost

    def _chain_llm_self_decompose():
        dec_msgs = prompt_decompose_compact(problem, "unknown")
        dec_text, dec_cost = caller.chat(dec_msgs, max_new_tokens=160, temperature=0.0)

        parsed_steps = []
        m = re.search(r'do\s*:\s*', dec_text, flags=re.I)
        sub = dec_text[m.end():] if m else dec_text
        for raw in (sub or "").splitlines():
            ln = re.sub(r'^\s*(?:\(?\d+\)?\s*[:.)-]\s*|step\s*\d+\s*[:.)-]\s*|[-*•]\s*)', '', raw.strip(), flags=re.I)
            if 2 <= len(ln.split()) <= 14:
                parsed_steps.append(ln.strip())
            if len(parsed_steps) >= 3:
                break
        steps_in = (parsed_steps + ["Compute the next needed count."] * 3)[:3]

        chain_msgs = prompt_chain_raw(problem, steps_in, require_step_ans=False)
        text, cost = caller.chat(chain_msgs, max_new_tokens=1000, temperature=0.0)

        for k in ("prompt_tokens", "completion_tokens", "latency_ms", "api_cost"):
            cost[k] = cost.get(k, 0.0) + dec_cost.get(k, 0.0)

        final = extract_final_line(text) or ""
        if is_csqa() and not final:
            final = extract_choice_letter(text) or ""
        return str(final), text, cost

    mode = (mode or "direct").lower()
    if mode == "direct":
        return _direct()
    if mode == "direct_steps":
        return _direct_steps(n_steps=3, require_step_ans=True)
    if mode == "chain":
        return _chain_llm_self_decompose()

    final, text, cost = _direct()
    if final.strip():
        return final, text, cost
    final2, text2, cost2 = _direct_steps()
    for k in cost:
        cost[k] = cost.get(k, 0.0) + cost2.get(k, 0.0)
    if final2.strip():
        return final2, text2, cost
    return _chain_llm_self_decompose()

# ----------------- 변경/추가 부분 시작 -----------------

def parse_redirect_map(raw: str) -> Dict[str, str]:
    """
    "--redirect_models 'src|dst,src2|dst2'" 같은 문자열을 dict로 파싱.
    """
    mp={}
    for part in (raw or "").split(","):
        part=part.strip()
        if not part: continue
        if "|" in part:
            src,dst = [x.strip() for x in part.split("|",1)]
            if src and dst: mp[src]=dst
    return mp

def compute_topsis(scores: List[Dict[str,float]], weights: Dict[str,float]) -> List[float]:
    """
    scores: [{acc, cost, tokens}, ...]
    weights: {'acc':w1,'cost':w2,'tokens':w3} (합=1 권장)
    acc는 benefit, cost/tokens는 cost criteria.
    """
    import numpy as np
    A = np.array([[s['acc'], s['cost'], s['tokens']] for s in scores], dtype=np.float64)

    # 벡터 정규화
    denom = np.sqrt((A**2).sum(axis=0)) + 1e-12
    R = A / denom

    # 가중
    w = np.array([weights.get('acc',0.5), weights.get('cost',0.25), weights.get('tokens',0.25)], dtype=np.float64)
    V = R * w

    # 이상/반이상해 (acc는 max가 이상해, cost/tokens는 min이 이상해)
    ideal_best  = np.array([V[:,0].max(), V[:,1].min(), V[:,2].min()])
    ideal_worst = np.array([V[:,0].min(), V[:,1].max(), V[:,2].max()])

    # 거리
    d_best  = np.sqrt(((V - ideal_best )**2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst)**2).sum(axis=1))

    # 근접도
    closeness = d_worst / (d_best + d_worst + 1e-12)
    return closeness.tolist()

def compute_extended_metrics(method_stats: Dict[str,Dict[str,Any]], baseline_key="baseline",
                             topsis_weights_str="acc:0.5,cost:0.15,tokens:0.35",
                             irt_alpha=0.8, irt_beta=0.2):
    """
    요약 출력 후 호출되어, 추가 지표들을 콘솔에 함께 출력.
    """
    def _wparse(s):
        d={}
        for part in (s or "").split(","):
            if ":" in part:
                k,v=part.split(":",1); d[k.strip()]=float(v.strip())
        ssum=sum(d.values()) or 1.0
        for k in d: d[k]/=ssum
        return d

    W=_wparse(topsis_weights_str)

    # 수집
    methods=list(method_stats.keys())
    base = method_stats[baseline_key]
    base_acc = (base["correct"]/base["total"]) if base["total"]>0 else 0.0
    base_cost= base.get("total_api_cost",0.0)
    base_tok = base.get("total_tokens",0)

    # 표준 행렬
    table=[]
    for m in methods:
        st=method_stats[m]
        acc=(st["correct"]/st["total"]) if st["total"]>0 else 0.0
        cost=st.get("total_api_cost",0.0)
        toks=st.get("total_tokens",0)
        table.append({"method":m,"acc":acc,"cost":cost,"tokens":float(toks)})

    # TOPSIS
    topsis_scores = compute_topsis(
        scores=[{"acc":r["acc"],"cost":r["cost"],"tokens":r["tokens"]} for r in table],
        weights=W
    )
    for i,r in enumerate(table):
        r["topsis"]=topsis_scores[i]

    # Cost Efficiency & 증가비
    for r in table:
        delta_acc  = r["acc"]  - base_acc
        delta_cost = r["cost"] - base_cost
        r["acc_gain"]   = delta_acc
        r["cost_gain"]  = delta_cost
        r["gain_ratio"] = (delta_acc / (delta_cost+1e-12)) if delta_cost>0 else float('inf')
        # cost per correct
        corr = max(1, int(round(r["acc"]*base["total"])))  # 동일 데이터 수에서의 correct 근사
        r["cpc"] = r["cost"]/corr
        base_corr = max(1, base["correct"])
        base_cpc  = base_cost/base_corr if base_corr>0 else float('inf')
        r["eff_vs_slm"] = (base_cpc / r["cpc"]) if r["cpc"]>0 else float('inf')

        # IRT-style reward (baseline 기준 차분)
        cost_norm = (r["cost"]/(base_cost+1e-12))
        r["irt_reward"] = irt_alpha*(r["acc"]-base_acc) - irt_beta*(cost_norm-1.0)

    # 출력
    print("\n=== Extended Metrics (vs SLM baseline) ===")
    print(f"- Baseline Acc={base_acc:.4f}, Cost=${base_cost:.6f}, Tokens={base_tok}")
    print(f"- TOPSIS weights = {W} (Acc↑ / Cost↓ / Tokens↓)")
    print(f"- IRT alpha={irt_alpha}, beta={irt_beta}  (Reward = a·ΔAcc - b·(Cost/Cost_SLM-1))\n")

    hdr=f"{'method':38s}  {'Acc':>6s}  {'$':>9s}  {'Tok':>9s}  {'ΔAcc':>7s}  {'Δ$':>9s}  {'ΔAcc/Δ$':>9s}  {'CPC_eff':>8s}  {'TOPSIS':>7s}  {'IRT_R':>7s}"
    print(hdr)
    print("-"*len(hdr))
    for r in sorted(table, key=lambda x:(-x["topsis"], -x["irt_reward"], -x["eff_vs_slm"], -x["acc"])):
        print(f"{r['method']:38s}  {r['acc']:6.4f}  {r['cost']:9.6f}  {int(r['tokens']):9d}  "
              f"{r['acc_gain']:7.4f}  {r['cost_gain']:9.6f}  "
              f"{(r['gain_ratio'] if r['gain_ratio']!=float('inf') else float('inf')):9.4f}  "
              f"{r['eff_vs_slm']:8.3f}  {r['topsis']:7.3f}  {r['irt_reward']:7.3f}")

# === [CHANGE] run_policy 시그니처에 slm_virtual_step_costs 추가 ===
def run_policy(problem:str, steps:List[str], base_parsed:List[Dict[str,Any]], base_final:str, bank, bm25, args, *,
               policy:str, router=None, model_names=None, la_bias=None, la_bias_disabled=False, device=None,
               callers:Dict[str,'HFChatCaller']=None, slm:'HFChatCaller'=None, rng:Optional[random.Random]=None,
               print_prefix:str="", slm_hint_chunks: Optional[List[str]] = None,
               slm_virtual_step_costs: Optional[List[Dict[str,float]]] = None,
               act_bias_delta: Optional[torch.Tensor] = None):

    N = len(steps)
    executed_chunks = [ (p.get("chunk") if p else "") for p in (base_parsed or []) ]  # 시작은 힌트=SLM 체인
    hint_chunks = slm_hint_chunks or executed_chunks

    # 힌트 기준 유효성
    hint_has = []; hint_isok = []; hint_vals=[]
    for k in range(N):
        has,isok,val = extract_ans_from_chunk(hint_chunks[k] or "", strict=False)
        hint_has.append(has); hint_isok.append(isok); hint_vals.append(val)

    chosen_actions=[]; chosen_models=[]
    final_answer=None; per_step_costs=[]
    total_tokens=0; total_api_cost=0.0
    modified_any=False  # Detract가 있을 때만 True로

    detract_pool=[m.strip() for m in args.detract_models.split(",") if m.strip()]
    escalate_pool=[m.strip() for m in args.escalate_models.split(",") if m.strip()]
    redirect_map = parse_redirect_map(getattr(args,"redirect_models",""))

    def _apply_redirect(name:str)->str: return redirect_map.get(name, name)
    def get_caller(name: str) -> 'HFChatCaller':
        if name not in callers: callers[name]=HFChatCaller(name, seed=args.seed, dtype=resolve_dtype(name))
        return callers[name]

    for t in range(N):
        # Router 입력은 힌트 텍스트 기준 (비용 X)
        units=["[CTX] "+problem]
        for k in range(t):
            units.append("[Q] "+steps[k]+"\n[A_SLM] "+(hint_chunks[k] or ""))
        q_text="[Q] "+steps[t]
        units.append(q_text+"\n[A_SLM] "+(hint_chunks[t] or ""))

        if policy=="router":
            slm_sig=slm_sig_from_text(hint_chunks[t] or "")
            action_pred, model_pred = predict_action_and_model(
                router, model_names, units, q_text, slm_sig, t,
                la_bias, args.disable_la_bias, device,
                act_bias_delta=act_bias_delta
            )

        elif policy=="random":
            probs=[0.55,0.30,0.15]; r=rng.random() if rng else random.random()
            action_pred=ACTIONS[0 if r<probs[0] else (1 if r<probs[0]+probs[1] else 2)]
            bigs=[m for m in model_names if m!="SLM"]; model_pred=(random.choice(bigs))
        else:
            raise ValueError("unknown policy")

        # 힌트가 비정상이면 Detract 강제
        forced=False
        if (not hint_has[t] or not hint_isok[t]):
            forced=True; action="Detract"
            chosen_model=detract_pool[0] if detract_pool else (model_pred if model_pred!="SLM" else "meta-llama/Llama-3.1-8B-Instruct")
        else:
            action=action_pred; chosen_model=model_pred

        if chosen_model=="SLM":
            pool=detract_pool if action=="Detract" else escalate_pool
            chosen_model=next((m for m in pool if m != args.slm_model), pool[0] if pool else args.slm_model)
        chosen_model=_apply_redirect(chosen_model)

        chosen_actions.append(action); chosen_models.append(chosen_model)

        # === 비용 집계: '가상 SLM 스텝 비용'을 실제 사용시에만 더함 ===
        slm_part = (slm_virtual_step_costs[t] if (slm_virtual_step_costs and t < len(slm_virtual_step_costs)) else {"prompt_tokens":0,"completion_tokens":0,"api_cost":0.0})

        if action=="Continue":
            # SLM(step t) 비용만 반영
            total_tokens += slm_part["prompt_tokens"] + slm_part["completion_tokens"]
            total_api_cost += slm_part["api_cost"]

            per_step_costs.append({
                "step": t+1, "action": "Continue", "model": "SLM",
                "cost": {"prompt_tokens": slm_part["prompt_tokens"], "completion_tokens": slm_part["completion_tokens"], "api_cost": slm_part["api_cost"]},
                "breakdown": {"slm": slm_part}
            })

            if print_prefix:
                info={"router_pred": action_pred if policy=="router" else "random",
                      "router_model": model_pred if policy=="router" else "random",
                      "forced_D": False, "value->": hint_vals[t]}
                print_decision_step(print_prefix, t, "Continue", "SLM",
                                    q_step=steps[t],
                                    slm_chunk_before=hint_chunks[t],
                                    edited_text=None, info=info)
            continue

        elif action=="Detract":
            caller=get_caller(chosen_model)
            supports=retrieve_support_pairs_bm25(bank, bm25, steps[t], args.ret_topk, args.ret_maxchars)
            orig_chunk = executed_chunks[t]
            new_chunk, has, isok, val, d_cost = run_detract_on_chunk(problem, steps[t], orig_chunk, caller, supports)
            executed_chunks[t]=new_chunk

            # 합산 (SLM + LLM)
            step_tok = slm_part["prompt_tokens"]+slm_part["completion_tokens"] + d_cost["prompt_tokens"]+d_cost["completion_tokens"]
            step_cost= slm_part["api_cost"] + d_cost["api_cost"]
            total_tokens += step_tok
            total_api_cost += step_cost
            modified_any=True  # Detract가 있으면 최종화 필요할 수 있음

            per_step_costs.append({
                "step": t+1, "action": "Detract", "model": chosen_model,
                "cost": {"prompt_tokens": slm_part["prompt_tokens"]+d_cost["prompt_tokens"],
                         "completion_tokens": slm_part["completion_tokens"]+d_cost["completion_tokens"],
                         "api_cost": step_cost},
                "breakdown": {"slm": slm_part, "llm": d_cost}
            })

            if print_prefix:
                info={"router_pred": action_pred if policy=='router' else 'random',
                      "router_model": model_pred if policy=='router' else 'random',
                      "forced_D": forced, "value->": val}
                print_decision_step(print_prefix, t, "Detract", chosen_model,
                                    q_step=steps[t],
                                    slm_chunk_before=hint_chunks[t],
                                    edited_text=new_chunk, info=info)

            # --- [NEW] Detract 이후에도 유효 ANS(숫자/문자) 없으면 즉시 Escalate ---
            if not isok:
                # 같은 스텝에서 바로 에스컬레이트 (SLM 가상 비용은 이미 한 번 더해졌으므로 여기선 LLM 비용만 추가)
                chosen_model2 = (_apply_redirect(escalate_pool[0]) if escalate_pool else chosen_model)
                caller2 = get_caller(chosen_model2)
                esc_final2, esc_cost2 = run_escalate_to_final(problem, steps, executed_chunks, t, caller2)

                # 비용: 이번 추가 Escalate는 LLM 비용만 합산 (slm_part는 이미 반영됨)
                total_tokens += esc_cost2["prompt_tokens"] + esc_cost2["completion_tokens"]
                total_api_cost += esc_cost2["api_cost"]

                # 히스토리/브레이크다운 기록 (같은 step에 두 번째 액션으로 기록)
                per_step_costs.append({
                    "step": t+1, "action": "Escalate", "model": chosen_model2,
                    "cost": {"prompt_tokens": esc_cost2["prompt_tokens"],
                            "completion_tokens": esc_cost2["completion_tokens"],
                            "api_cost": esc_cost2["api_cost"]},
                    "breakdown": {"llm": esc_cost2}
                })
                chosen_actions.append("Escalate")
                chosen_models.append(chosen_model2)

                if print_prefix:
                    info2 = {"router_pred": action_pred if policy=='router' else 'random',
                            "router_model": model_pred if policy=='router' else 'random',
                            "forced_D": forced, "value->": "N/A (Detract failed; forced Escalate)"}
                    print_decision_step(print_prefix, t, "Escalate", chosen_model2,
                                        q_step=steps[t], slm_chunk_before=executed_chunks[t],
                                        edited_text=None, info=info2)

                if esc_final2:
                    final_answer = esc_final2
                break
            # --- [NEW] 끝 ---


        elif action=="Escalate":
            caller=get_caller(chosen_model)
            esc_final, esc_cost = run_escalate_to_final(problem, steps, executed_chunks, t, caller)

            # 합산 (SLM + LLM)
            step_tok = slm_part["prompt_tokens"]+slm_part["completion_tokens"] + esc_cost["prompt_tokens"]+esc_cost["completion_tokens"]
            step_cost= slm_part["api_cost"] + esc_cost["api_cost"]
            total_tokens += step_tok
            total_api_cost += step_cost

            per_step_costs.append({
                "step": t+1, "action": "Escalate", "model": chosen_model,
                "cost": {"prompt_tokens": slm_part["prompt_tokens"]+esc_cost["prompt_tokens"],
                         "completion_tokens": slm_part["completion_tokens"]+esc_cost["completion_tokens"],
                         "api_cost": step_cost},
                "breakdown": {"slm": slm_part, "llm": esc_cost}
            })

            if print_prefix:
                info={"router_pred": action_pred if policy=='router' else 'random',
                      "router_model": model_pred if policy=='router' else 'random',
                      "forced_D": forced, "value->": hint_vals[t]}
                print_decision_step(print_prefix, t, "Escalate", chosen_model,
                                    q_step=steps[t],
                                    slm_chunk_before=hint_chunks[t],
                                    edited_text=None, info=info)
            if esc_final:
                final_answer = esc_final
            break

    # (Detract가 있었을 때만) 최종화 시도
    if (not final_answer) and modified_any:
        finalizer_pool = ([m for m in args.escalate_models.split(",") if m.strip()] + [m for m in args.detract_models.split(",") if m.strip()])
        finalizer_name = (finalizer_pool[0] if finalizer_pool else args.slm_model).strip()
        finalizer_name = _apply_redirect(finalizer_name)
        caller = callers.get(finalizer_name) or HFChatCaller(finalizer_name, seed=args.seed, dtype=resolve_dtype(finalizer_name))
        callers[finalizer_name] = caller
        if getattr(args,"router_finalize_mode","auto")=="auto":
            fin, fcost, _dbg = run_finalize_router_auto(problem, steps, executed_chunks, caller, max_new_tokens=80)
        else:
            fin, fcost, _ = run_finalize(problem, steps, executed_chunks, caller, max_new_tokens=80, mode=args.router_finalize_mode)
        if fin: final_answer=fin
        total_tokens += fcost.get("prompt_tokens",0)+fcost.get("completion_tokens",0)
        total_api_cost += fcost.get("api_cost",0.0)
        if print_prefix:
            print("- Finalize ---------------------------------")
            print(f"  model={finalizer_name} FINAL: {final_answer}")

    if not final_answer:
        final_answer = base_final  # Escalate도 Detract도 없으면 baseline FINAL 사용

    return {"actions":chosen_actions,"models":chosen_models,"executed_steps":executed_chunks,
            "final":final_answer,"correct":False,"step_costs":per_step_costs,
            "total_api_cost":total_api_cost,"total_tokens":total_tokens}

# ---- main ----
# ---- main ----
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k","csqa","openbookqa","race","race-middle","race-high"], default="gsm8k")
    ap.add_argument("--split", default=None)
    ap.add_argument("--router_ckpt", default="ckpts/router_rl.pt")
    ap.add_argument("--slm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--llm_only_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llm_only_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--detract_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--escalate_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--bank", default="bank_ret.jsonl")
    ap.add_argument("--ret_topk", type=int, default=1)
    ap.add_argument("--ret_maxchars", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1319)
    ap.add_argument("--print_n", type=int, default=20)
    ap.add_argument("--show_slm", action="store_true")
    ap.add_argument("--show_retrieval", action="store_true")
    ap.add_argument("--out", default="predictions_router_compact_llmDirectstep_profileslm.jsonl")
    ap.add_argument("--disable_la_bias", action="store_true")
    ap.add_argument("--free_models", type=str, default="")
    ap.add_argument("--fp32_models", type=str, default="")
    ap.add_argument("--llm_only_mode", choices=["direct","direct_steps","chain","auto"], default="direct_steps")
    ap.add_argument("--finalize_mode", choices=["attach_direct","from_steps","direct"], default="attach_direct")
    ap.add_argument("--router_finalize_mode", choices=["attach_direct","from_steps","direct","auto"], default="auto")
    ap.add_argument("--swap_models", type=str, default="")
    ap.add_argument("--llm_only_profile", choices=["default","slm"], default="slm")

    # [NEW] 실행 제어 플래그들
    ap.add_argument("--no_llm_only", action="store_true",
                    help="LLM-only baselines 비활성화")
    ap.add_argument("--no_random", action="store_true",
                    help="Random policy 비활성화")
    ap.add_argument("--router_only", action="store_true",
                    help="Router만 실행 (SLM baseline은 힌트/비교용으로만 유지)")

    # [추가] 비용/지표 제어 옵션
    ap.add_argument("--charge_slm_to_router", action="store_true", default=True,
                    help="Router/Random에 SLM decompose + SLM chain 비용을 합산")
    ap.add_argument("--redirect_models", type=str, default="",
                    help='예: "Qwen/Qwen2.5-14B-Instruct|Qwen/Qwen2.5-7B-Instruct"')
    ap.add_argument("--topsis_weights", type=str, default="acc:0.5,cost:0.15,tokens:0.35",
                    help='예: "acc:0.45,cost:0.15,tokens:0.40" (합은 자동 정규화)')
    ap.add_argument("--irt_alpha", type=float, default=0.8)
    ap.add_argument("--irt_beta", type=float, default=0.2)

    ap.add_argument("--act_bias", type=str, default="",
                help='Additive action bias "c,d,e" for (Continue,Detract,Escalate). Example: "-0.2,0.0,+0.3"')


    args=ap.parse_args()

    # task set
    set_task(args.task)

    # [NEW] router_only면 나머지 비활성화
    if args.router_only:
        args.no_llm_only = True
        args.no_random   = True

    if args.free_models:
        for m in args.free_models.split(","):
            m=m.strip()
            if m: FREE_MODELS.add(m.lower())
    register_fp32_models(args.fp32_models)

    llm_models=[s.strip() for s in args.llm_only_models.split(",") if s.strip()]
    if not llm_models:
        llm_models=[args.llm_only_model.strip()]

    set_seed(args.seed); rng=random.Random(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # router, model_names, la_bias = load_router(args.router_ckpt, device)
    router, model_names, la_bias, ablate_flags = load_router(args.router_ckpt, device)
    router._ablate_forbid = ("forbid" in ablate_flags)
    router._ablate_la     = ("la" in ablate_flags)
    # la를 껐다면 disable 플래그 올려서 위에서 la_bias 적용 안 하게
    if router._ablate_la:
        args.disable_la_bias = True

    act_bias_delta = parse_act_bias(args.act_bias, device)

    # optional slot swap
    if args.swap_models:
        def _find_idx(name: str) -> int:
            for i,s in enumerate(model_names):
                if s.lower()==name.lower(): return i
            raise ValueError(f'"{name}" not in router model_names: {model_names}')
        try:
            a_raw,b_raw=[s.strip() for s in args.swap_models.split("|",1)]
            ia,ib=_find_idx(a_raw),_find_idx(b_raw)
            if 0 in (ia,ib): print("[WARN] Not swapping SLM (slot 0). Ignored.")
            else:
                model_names[ia],model_names[ib]=model_names[ib],model_names[ia]
                print(f"[INFO] Swapped router slots: {a_raw} <-> {b_raw}")
        except Exception as e:
            print(f"[WARN] swap_models failed: {e}")

    callers: Dict[str, HFChatCaller] = {}
    def get_caller(name: str, *, dtype: Optional[str]=None, profile: str="default") -> HFChatCaller:
        if name not in callers:
            callers[name]=HFChatCaller(name, seed=args.seed, dtype=(dtype or resolve_dtype(name)), profile=profile)
        return callers[name]

    # SLM은 항상 필요 (분해/체인 힌트 및 baseline)
    slm=get_caller(args.slm_model, dtype="float32", profile="slm")

    # [CHANGE] LLM-only 끄면 사전 로딩 생략
    if not args.no_llm_only:
        for m in llm_models:
            get_caller(m, dtype=resolve_dtype(m), profile=args.llm_only_profile)

    bank=load_case_bank_jsonl(args.bank); bm25=build_bm25_from_bank(bank) if bank else None

    # --- dataset loading (task-aware) ---
    if args.task == "gsm8k":
        ds_split = args.split or "test"
        ds_all = load_dataset("gsm8k","main")[ds_split]

    elif args.task == "csqa":
        ds_split = args.split or "validation"
        ds_all = load_dataset("commonsense_qa")[ds_split]

    elif args.task == "openbookqa":
        ds_split = args.split or "test"
        ds_all = load_dataset("openbookqa", "main")[ds_split]

    elif args.task in ("race","race-middle","race-high"):
        cfg = "all" if args.task=="race" else ("middle" if args.task=="race-middle" else "high")
        ds_split = args.split or "test"   # RACE는 validation/test 모두 존재
        ds_all = load_dataset("race", cfg)[ds_split]



    start=max(0,args.start)
    end=len(ds_all) if args.limit<=0 else min(len(ds_all), start+args.limit)

    if args.out == "predictions_router_compact_llmDirectstep_profileslm.jsonl":
        args.out = f"predictions_{args.task}_router_compact.jsonl"

    # [NEW] methods를 플래그에 맞게 구성
    methods = ["baseline", "router"]
    if not args.no_random:
        methods.append("random")
    if not args.no_llm_only:
        methods += [f"llm_only::{m}" for m in llm_models]

    method_stats={m:{"correct":0,"total":0,"action_hist":{a:0 for a in ACTIONS},
                     "model_hist":{},"total_tokens":0,"total_api_cost":0.0} for m in methods}

    dump=open(args.out,"w",encoding="utf-8")
    printed = 0

    for i in tqdm(range(start,end), desc=f"Router inference ({args.task}/{ds_split})"):
        ds = ds_all[i]

        if args.task == "gsm8k":
            q = ds["question"].strip()
            gold_full = ds["answer"]
            gold = last_number(gold_full) or gold_full.strip()
            problem_text = q

        elif args.task == "csqa":
            stem = ds["question"].strip()
            labels = ds["choices"]["label"]
            texts  = ds["choices"]["text"]
            options = "\n".join([f"{lab}. {txt}" for lab, txt in zip(labels, texts)])
            gold = ds["answerKey"].strip().upper()[:1]
            # 레이블 범위 동적 안내(보통 A–E)
            maxlab = (sorted(set(labels)) or ["E"])[-1]
            problem_text = (
                f"{stem}\n\nOptions:\n{options}\n\n"
                f"Answer with the option letter (A-{maxlab})."
            )

        elif args.task == "openbookqa":
            stem = ds["question_stem"].strip()
            labels = ds["choices"]["label"]    # 보통 ['A','B','C','D']
            texts  = ds["choices"]["text"]
            options = "\n".join([f"{lab}. {txt}" for lab, txt in zip(labels, texts)])
            gold = ds["answerKey"].strip().upper()[:1]
            maxlab = (sorted(set(labels)) or ["D"])[-1]
            problem_text = (
                f"{stem}\n\nOptions:\n{options}\n\n"
                f"Answer with the option letter (A-{maxlab})."
            )

        elif args.task in ("race","race-middle","race-high"):
            # RACE: article + question + options(list) + answer
            passage = ds["article"].strip()
            stem    = ds["question"].strip()
            opts    = ds["options"]           # list[str], length 4
            labels  = ["A","B","C","D"][:len(opts)]
            options = "\n".join([f"{lab}. {txt}" for lab, txt in zip(labels, opts)])
            gold    = ds["answer"].strip().upper()[:1]  # already A–D

            problem_text = (
                f"Passage:\n{passage}\n\n"
                f"Question:\n{stem}\n\n"
                f"Options:\n{options}\n\n"
                f"Answer with the option letter (A-{labels[-1]})."
            )

        dbg = bool(args.print_n and printed < args.print_n)
        if dbg:
            print_case_header(i, problem_text, gold)

        # 1) Decompose (SLM)
        deco_text, deco_cost = slm.chat(prompt_decompose_compact(problem_text,"unknown"), max_new_tokens=160, temperature=0.0)
        deco_tokens = deco_cost["prompt_tokens"] + deco_cost["completion_tokens"]
        deco_api    = deco_cost["api_cost"]

        # parse 3 steps
        steps=[]
        m=re.search(r'do\s*:\s*', deco_text, flags=re.I)
        sub=deco_text[m.end():] if m else deco_text
        for raw in (sub or "").splitlines():
            ln=raw.strip()
            if not ln: continue
            ln=re.sub(r'^\s*(?:\(?\d+\)?\s*[:.)-]\s*|step\s*\d+\s*[:.)-]\s*|[-*•]\s*)','',ln, flags=re.I)
            if 2<=len(ln.split())<=14: steps.append(ln.strip())
            if len(steps)>=3: break
        if len(steps)<3: steps=(steps+["Compute the next needed count."]*3)[:3]

        # 2) Baseline(SLM)
        base_parsed, base_final, base_text, base_cost = run_singlepass_chain(slm, problem_text, steps)
        base_tokens = base_cost["prompt_tokens"] + base_cost["completion_tokens"]
        base_api    = base_cost["api_cost"]
        base_ok = judge_answer(gold, base_final or "")
        if dbg and args.show_slm:
            print_slm_debug(steps, base_text, base_parsed)

        # baseline 집계
        method_stats["baseline"]["total"] += 1
        if base_ok: method_stats["baseline"]["correct"] += 1
        method_stats["baseline"]["total_tokens"] += (deco_tokens + base_tokens)
        method_stats["baseline"]["total_api_cost"] += (deco_api + base_api)

        # 3) SLM 가상 스텝 비용 분배
        slm_virtual_step_costs = build_virtual_slm_step_costs(
            base_parsed, base_cost, args.slm_model, slm.tokenizer, prompt_share_mode="equal"
        )

        # 4) Router
        router_out = run_policy(
            problem_text, steps, base_parsed, base_final, bank, bm25, args,
            policy="router", router=router, model_names=model_names,
            la_bias=la_bias, la_bias_disabled=args.disable_la_bias,
            device=device, callers=callers, slm=slm, rng=rng,
            print_prefix=("router" if dbg else ""),
            slm_hint_chunks=[p["chunk"] for p in base_parsed],
            slm_virtual_step_costs=slm_virtual_step_costs,
            act_bias_delta=act_bias_delta
        )
        if args.charge_slm_to_router:
            router_out["total_tokens"]  += deco_tokens
            router_out["total_api_cost"]+= deco_api

        router_ok = judge_answer(gold, router_out["final"] or "")
        router_out["correct"] = bool(router_ok)
        st=method_stats["router"]; st["total"]+=1
        if router_ok: st["correct"]+=1
        st["total_tokens"]+=router_out["total_tokens"]; st["total_api_cost"]+=router_out["total_api_cost"]
        for a in router_out["actions"]: st["action_hist"][a]=st["action_hist"].get(a,0)+1
        for mname in router_out["models"]: st["model_hist"][mname]=st["model_hist"].get(mname,0)+1

        # 5) Random (옵션)
        random_out = None
        if not args.no_random:
            random_out = run_policy(
                problem_text, steps, base_parsed, base_final, bank, bm25, args,
                policy="random", router=None, model_names=model_names,
                la_bias=None, la_bias_disabled=True,
                device=device, callers=callers, slm=slm, rng=rng,
                print_prefix=("random" if dbg else ""),
                slm_hint_chunks=[p["chunk"] for p in base_parsed],
                slm_virtual_step_costs=slm_virtual_step_costs
            )
            if args.charge_slm_to_router:
                random_out["total_tokens"]  += deco_tokens
                random_out["total_api_cost"]+= deco_api

            random_ok = judge_answer(gold, random_out["final"] or "")
            random_out["correct"]=bool(random_ok)
            st=method_stats["random"]; st["total"]+=1
            if random_ok: st["correct"]+=1
            st["total_tokens"]+=random_out["total_tokens"]; st["total_api_cost"]+=random_out["total_api_cost"]
            for a in random_out["actions"]: st["action_hist"][a]=st["action_hist"].get(a,0)+1
            for mname in random_out["models"]: st["model_hist"][mname]=st["model_hist"].get(mname,0)+1

        # 6) LLM-only baselines (옵션)
        llm_only_results={}
        llm_results_for_print=[]
        if not args.no_llm_only:
            for mname in llm_models:
                final, raw, cost = run_llm_only(callers[mname], problem_text, None, mode=args.llm_only_mode)
                ok = judge_answer(gold, final or "")
                llm_only_results[mname] = {"final": final, "correct": bool(ok)}
                tag = f"llm_only::{mname}"
                st = method_stats[tag]; st["total"] += 1
                if ok: st["correct"] += 1
                st["total_tokens"] += cost["prompt_tokens"] + cost["completion_tokens"]
                st["total_api_cost"] += cost["api_cost"]
                llm_results_for_print.append((mname, {"final": final, "correct": bool(ok)}))

        # 7) per-problem dump
        rec={"id": f"{args.task}-{ds_split}-{i:05d}",
             "problem": problem_text,
             "steps": steps,
             "baseline":{"final": base_final,"correct": bool(base_ok),"raw_chain": base_text,"parsed": base_parsed,
                         "cost":{"decompose": deco_cost,"chain": base_cost,
                                 "total_tokens": (deco_tokens + base_tokens),
                                 "total_api_cost": (deco_api + base_api)}},
             "router": router_out,
             "gold": str(gold)}
        if random_out is not None:
            rec["random"] = random_out
        if not args.no_llm_only:
            rec["llm_only"] = llm_only_results

        dump.write(json.dumps(rec, ensure_ascii=False)+"\n")

        # 8) 디버그 출력
        if dbg:
            if args.no_llm_only and args.no_random:
                # 순수 Router만
                print("- Results --------------------------------")
                print(f"Baseline FINAL={base_final}  -> {'OK' if base_ok else 'WRONG'}")
                print(f"Router   FINAL={router_out['final']} -> {'OK' if router_out['correct'] else 'WRONG'}")
            else:
                rand_final = random_out["final"] if random_out is not None else "-"
                rand_ok    = random_out["correct"] if random_out is not None else False
                print_result_multi(base_final, base_ok,
                                   llm_results_for_print,
                                   router_out["final"], router_out["correct"],
                                   rand_final, rand_ok)
            printed += 1

    dump.close()

    print("\n=== Summary by Method ===")
    for m in methods:
        st=method_stats[m]; acc=(st["correct"]/st["total"]) if st["total"]>0 else 0.0
        print(f"[{m}] Acc: {acc:.4f} ({st['correct']}/{st['total']})")
        print(f"    Total tokens : {st['total_tokens']}")
        print(f"    Total api $  : {st['total_api_cost']:.6f}")
        print(f"    [action distribution]")
        for a in ACTIONS: print(f"      {a:9s}: {st['action_hist'].get(a,0)}")
        print(f"    [model picks]")
        for name,cnt in st["model_hist"].items(): print(f"      {name}: {cnt}")
        print()

    # ---- 추가 지표 출력
    compute_extended_metrics(method_stats, baseline_key="baseline",
                             topsis_weights_str=args.topsis_weights,
                             irt_alpha=args.irt_alpha, irt_beta=args.irt_beta)

    print(f"\nPer-problem dump -> {args.out}")

    for c in list(callers.values()):
        try: c.close()
        except: pass

if __name__ == "__main__":
    main()