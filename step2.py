#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Router-driven inference (clean build)

Decisions:
- Removed IRT/TOPSIS/extra diagnostics.
- Removed bias knobs (no la_bias / act_bias).
- Defaults: --charge_slm_to_router (ON), LLM-only uses chain mode with profile='slm'.
- Methods tracked: baseline (SLM), router, LLM-only.
"""

import os, re, json, math, random, argparse, gc, time
from typing import List, Dict, Any, Tuple, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoConfig, AutoModel,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM
)
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from datasets import load_dataset
from tqdm import tqdm

# =========================
# small utilities
# =========================
ACTIONS = ["Continue", "Detract", "Escalate"]
NUM_WITH_COMMAS = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
CHOICE_LETTER_RE = re.compile(r'\b([A-J])\b', flags=re.I)

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def last_number(text: str) -> Optional[str]:
    if not text: return None
    toks = NUM_WITH_COMMAS.findall(text)
    if not toks: return None
    return toks[-1].replace(",", "")

def extract_final_line(text: str) -> Optional[str]:
    if not text: return None
    for p in [r'^\s*FINAL(?:\s*ANSWER)?\s*[:=]\s*(.+?)\s*$',
              r'^\s*Answer\s*[:=]\s*(.+?)\s*$',
              r'^\s*Result\s*[:=]\s*(.+?)\s*$']:
        m = re.search(p, text, flags=re.I|re.M)
        if m: return m.group(1).strip()
    return None

def extract_choice_letter(text: str) -> Optional[str]:
    if not text: return None
    m = CHOICE_LETTER_RE.search(text)
    return m.group(1).upper() if m else None

def judge_numeric(gold: str, pred: str) -> bool:
    def _pick(s):
        m = NUM_WITH_COMMAS.findall(str(s))
        return m[-1].replace(",","") if m else None
    g, p = _pick(gold), _pick(pred)
    return (g==p) if (g is not None and p is not None) else (str(gold).strip()==str(pred).strip())

TASK = "gsm8k"
def set_task(t: str):  # supports gsm8k/csqa/openbookqa/race
    global TASK
    TASK = (t or "gsm8k").lower()

def is_csqa() -> bool:
    return TASK in {"csqa", "openbookqa", "race", "race-middle", "race-high"}

def judge_answer(gold: str, pred: str) -> bool:
    if is_csqa():
        g = (gold or "").strip().upper()[:1]
        p = (pred or "").strip()
        p = extract_choice_letter(p) or p[:1].upper()
        return bool(g) and (p == g)
    else:
        return judge_numeric(gold, pred)

# =========================
# simple API-cost model
# =========================
MODEL_COST_CONFIG = {
    'qwen25_1p5b_instruct':  {'input_cost': 0.05e-6, 'output_cost': 0.10e-6},
    'qwen25_7b_instruct':    {'input_cost': 0.10e-6, 'output_cost': 0.20e-6},
    'qwen25_14b_instruct':   {'input_cost': 0.15e-6, 'output_cost': 0.30e-6},
    'llama31_8b_instruct':   {'input_cost': 0.10e-6, 'output_cost': 0.20e-6},
}
FREE_MODELS = set()
FORCE_DTYPE: Dict[str, str] = {}

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
    else:
        cost['api_cost']=0.0

def register_fp32_models(fp32_list: str):
    FORCE_DTYPE.clear()
    for m in (fp32_list or "").split(","):
        mm = m.strip()
        if mm: FORCE_DTYPE[mm.lower()] = "float32"

def resolve_dtype(model_name: str, default: str = "float16") -> str:
    return FORCE_DTYPE.get(model_name.lower(), default)

# =========================
# HF caller (chat)
# =========================
class HFChatCaller:
    def __init__(self, model_name: str, dtype: Optional[str]="float16",
                 seed: Optional[int]=None, hf_token: Optional[str]=None, profile: str="default"):
        self.model_name, self.profile, self.seed = model_name, (profile or "default").lower(), seed
        if torch.cuda.is_available():
            dt = {"bfloat16":torch.bfloat16, "bf16":torch.bfloat16, "float32":torch.float32, "fp32":torch.float32}.get(str(dtype).lower(), torch.float16)
        else:
            dt = torch.float32
        tok = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        auth = ({'token': tok, 'use_auth_token': tok} if tok else {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **auth)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **auth)
        fm_kwargs = dict(trust_remote_code=True, torch_dtype=dt, device_map="auto",
                         low_cpu_mem_usage=True, attn_implementation="eager", **auth)
        self.model = (AutoModelForSeq2SeqLM if getattr(cfg,"is_encoder_decoder",False)
                      else AutoModelForCausalLM).from_pretrained(model_name, **fm_kwargs)
        self.model.eval()

    def _apply_template(self, messages: List[Dict[str,str]]) -> str:
        if isinstance(messages, str):
            messages = [{"role":"user","content":messages}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            lines=[f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages]; lines.append("ASSISTANT:")
            return "\n".join(lines)

    def _gen_once(self, inputs, eos_id, pad_id, *, do_sample: bool, temperature: Optional[float],
                  top_p: Optional[float], top_k: Optional[int],
                  repetition_penalty: float, no_repeat_ngram_size: int, max_new_tokens: int):
        try:
            gcfg = self.model.generation_config
            gcfg.do_sample=False; gcfg.temperature=None; gcfg.top_p=None; gcfg.top_k=None
        except: pass
        lp = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
        kwargs = dict(
            max_new_tokens=int(max_new_tokens), min_new_tokens=1,
            return_dict_in_generate=True, output_scores=False, use_cache=True,
            eos_token_id=eos_id, pad_token_id=pad_id, logits_processor=lp, renormalize_logits=True,
            do_sample=do_sample, temperature=(float(temperature) if do_sample else None),
            top_p=(float(top_p) if do_sample else None),
            top_k=(int(top_k) if do_sample and top_k is not None else None),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )
        if self.profile == "slm":
            kwargs.update(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        return out, (time.perf_counter()-t0)*1000.0

    def chat(self, messages: List[Dict[str,str]], *, max_new_tokens: int = 512, temperature: float = 0.0,
             top_p: float = 0.95, top_k: Optional[int] = None, sample: Optional[bool] = None,
             repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0) -> Tuple[str, Dict[str,float]]:
        prompt = self._apply_template(messages)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        enc = self.tokenizer(prompt, return_tensors="pt")
        dev = next(self.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        eos_id = getattr(self.model.config, "eos_token_id", None) or self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        use_sample = sample if sample is not None else (float(temperature) > 0.0 and self.profile != "slm")

        out, lat = self._gen_once(
            enc, eos_id, pad_id,
            do_sample=use_sample, temperature=temperature if use_sample else None,
            top_p=top_p if use_sample else None, top_k=top_k if use_sample else None,
            repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
        seq = out.sequences[0]
        plen = int(enc["input_ids"].shape[-1])
        gen_ids = seq if getattr(self.model.config,"is_encoder_decoder",False) else seq[plen:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        cost = {
            "prompt_tokens": int(enc["input_ids"].shape[-1]),
            "completion_tokens": int(gen_ids.numel()),
            "latency_ms": float(lat),
        }
        apply_api_cost(cost, self.model_name)
        return text, cost

    def close(self):
        try: del self.model; del self.tokenizer
        except: pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# =========================
# prompts (minimal)
# =========================
def prompt_decompose_compact(problem:str, qtype:str, n_steps:int=3)->List[Dict[str,str]]:
    if is_csqa():
        user = f"""Decompose the MCQA into EXACTLY {n_steps} short steps (≤10 words each). Do not answer.

Question:
{problem}

Answer Format:
To solve "xxx", do:
1. ...
2. ...
3. ..."""
        return [
            {"role":"system","content":"Break into exactly 3 short decision steps. No answer."},
            {"role":"user","content":user},
        ]
    else:
        user = f"""Decompose the math problem into EXACTLY {n_steps} concise numeric steps.

Problem:
{problem}

Answer Format:
To solve "xxx", do:
1. ...
2. ...
3. ..."""
        return [
            {"role":"system","content":"Break into exactly 3 atomic numeric steps."},
            {"role":"user","content":user},
        ]

def prompt_chain_raw(problem:str, steps:List[str])->List[Dict[str,str]]:
    s = "\n".join([f"Step {i+1}: {steps[i]}" for i in range(len(steps))])
    if is_csqa():
        user = f"""Solve by following the steps and output the final letter.

Question:
{problem}

Steps to follow:
{s}

Rules (STRICT):
- For EACH step, write one short line ending with: ANS: <letter>.
- Then output exactly: FINAL: <letter>."""
        return [
            {"role":"system","content":"Solve MCQA step-by-step and end with 'FINAL: <letter>'."},
            {"role":"user","content":user},
        ]
    else:
        user = f"""Solve by following the steps and end with a numeric 'FINAL:'.

Problem:
{problem}

Steps to follow:
{s}

Rules:
- For each step, provide a brief calculation line.
- End with: FINAL: <number>"""
        return [
            {"role":"system","content":"Solve math step-by-step and end with 'FINAL:'."},
            {"role":"user","content":user},
        ]

def prompt_llm_only(problem:str)->List[Dict[str,str]]:
    if is_csqa():
        return [
            {"role":"system","content":"Answer with a single option letter."},
            {"role":"user","content":f"Answer this MCQA with one line: FINAL: <letter>\n\nQuestion:\n{problem}"}
        ]
    else:
        return [
            {"role":"system","content":"You are a careful math solver."},
            {"role":"user","content":f"Solve the problem and end with exactly one line 'FINAL: <number>'.\n\nProblem:\n{problem}"}
        ]

def prompt_edit_raw_chunk_minimal(problem:str, subproblem:str, current_chunk:str)->List[Dict[str,str]]:
    if is_csqa():
        user = f"""Edit this ONE step of an MCQA.

Return EXACTLY TWO lines:
Line 1: short reason; quote a keyword from the chosen option.
Line 2: ANS: <letter>

Question:
{problem}

Subproblem:
{subproblem}

Current:
\"\"\"{(current_chunk or '').strip()[:600]}\"\"\""""
        return [{"role":"user","content":user}]
    else:
        user=f"""Edit minimally for THIS subproblem.

Return EXACTLY TWO lines:
Line 1: one short sentence solving it.
Line 2: ANS: <number>

Problem:
{problem}

Subproblem:
{subproblem}

Current:
\"\"\"{(current_chunk or '').strip()[:600]}\"\"\""""
        return [{"role":"user","content":user}]

def prompt_force_final_only(problem:str, prev_text:str)->List[Dict[str,str]]:
    tag = "<letter>" if is_csqa() else "<number>"
    user=f"""You wrote:

\"\"\"{(prev_text or '').strip()[:1000]}\"\"\"

Now output EXACTLY ONE LINE:
FINAL: {tag}"""
    return [{"role":"user","content":user}]

# =========================
# parsing helpers
# =========================
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

def extract_ans_from_chunk(chunk: str, *, strict: bool = False) -> Tuple[bool, bool, str]:
    if not chunk: return False, False, ""
    text=chunk.strip()
    m = re.search(r'\bANS(?:WER)?\s*[:=]\s*([^\n]+)', text, flags=re.I)
    if m:
        val_raw=m.group(1).strip()
        if not is_csqa():
            num=last_number(val_raw)
            if num: return True, True, num
        else:
            let=extract_choice_letter(val_raw)
            if let: return True, True, let
        return True, False, val_raw
    if strict: return False, False, ""
    if is_csqa():
        let=extract_choice_letter(text)
        if let: return True, True, let
    m2 = re.search(r'=\s*[\$]?\s*(-?\d[\d,]*(?:\.\d+)?)\b', text)
    if m2: return True, True, m2.group(1).replace(",","")
    return False, False, ""

def slm_sig_from_text(text: str) -> np.ndarray:
    s=(text or ""); chars=list(s); L=max(1,len(chars))
    return np.array([
        1.0 if s.strip() else 0.0,
        1.0 if re.search(r'-?\d+(?:\.\d+)?', s) else 0.0,
        sum(c.isdigit() for c in chars)/L,
        sum(c in "+-*/=^" for c in chars)/L,
        float(len(s.split()))
    ], dtype=np.float32)

# =========================
# tiny retrieval (optional)
# =========================
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

# =========================
# Router model (inference)
# =========================
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
        with torch.no_grad(): out=self.bert(**enc); pooled= self.pool(out.last_hidden_state, enc["attention_mask"])
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
            if isinstance(x,(list,tuple)): return torch.tensor(x, device=device, dtype=torch.float32) if len(x)>0 else torch.full((d,), fill, device=device, dtype=torch.float32)
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
        if self._aux_dim == 0: aux = aux.new_zeros((aux.size(0), 0))
        elif aux.size(1) > self._aux_dim: aux = aux[:, :self._aux_dim]
        elif aux.size(1) < self._aux_dim: aux = torch.cat([aux, aux.new_zeros((aux.size(0), self._aux_dim-aux.size(1)))], dim=1)
        aux=(aux-self.aux_mean)/self.aux_std.clamp(min=1e-6); slm_sigs=(slm_sigs_in.float()-self.slm_mean)/self.slm_std.clamp(min=1e-6)
        stepv=self.step_emb(step_idx.clamp(min=0, max=self.step_emb.num_embeddings-1))
        cur_feat=torch.cat([q_emb,past_mean,sim_max,sim_min,sim_ent,aux,slm_sigs,stepv],dim=1)
        cur_h=self.cur_mlp(cur_feat)
        fused=self.fuse(torch.cat([h_last,cur_h],dim=1)); fused=self.ln_fuse(self.dropout(fused))
        return self.act_head(fused), self.mdl_head(fused)

def forbid_slm_on_de(mdl_logits: torch.Tensor, act_ids: torch.Tensor, slm_idx: int = 0):
    out=mdl_logits.clone()
    mask=(act_ids!=0)
    if mask.any(): out[mask, slm_idx]=out.new_tensor(-1e9)
    return out

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
    aux_dim = len(fs["aux_mean"]) if "aux_mean" in fs else int(cfg.get("aux_dim", 6))
    slm_dim = len(fs["slm_mean"]) if "slm_mean" in fs else int(cfg.get("slm_sig_dim", 5))

    model = GRURouter(
        num_models=len(model_names),
        aux_dim=aux_dim, slm_sig_dim=slm_dim,
        bert_name=bert_name, max_len=max_len, freeze_bert=freeze_bert
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

    model.to(device).eval()
    return model, model_names

def _fit_vec_to_dim(t: torch.Tensor, target_dim: int) -> torch.Tensor:
    if t.dim() == 1: t = t.unsqueeze(0)
    B,Din = t.size(0), t.size(1)
    if target_dim == Din: return t
    if target_dim == 0:  return t.new_zeros((B,0))
    if Din == 0:         return t.new_zeros((B,target_dim))
    if Din > target_dim: return t[:, :target_dim]
    pad = t.new_zeros((B, target_dim-Din))
    return torch.cat([t, pad], dim=1)

def predict_action_and_model(router: GRURouter, model_names: List[str], units: List[str], q_text: str,
                             slm_sig_vec: np.ndarray, step_idx: int, device):
    with torch.no_grad():
        slm_sig_t = torch.from_numpy(slm_sig_vec).float().to(device).unsqueeze(0)
        slm_sig_t = _fit_vec_to_dim(slm_sig_t, router._slm_dim)
        act_logits, mdl_logits = router([units], [q_text], slm_sig_t, torch.tensor([step_idx], dtype=torch.long, device=device))
        pa = int(act_logits.argmax(1).item())
        blended = forbid_slm_on_de(mdl_logits, torch.tensor([pa], device=device), slm_idx=0)
        pm = int(blended.argmax(1).item())
        if pa != 0 and pm == 0 and blended.size(1) > 1:
            vals = blended[0].detach().cpu().numpy()
            order = np.argsort(-vals)
            for idx in order:
                if idx != 0: pm = int(idx); break
        return ACTIONS[pa], model_names[pm]

# =========================
# SLM single-pass + cost split
# =========================
def _count_tokens_tok(tok, text: str) -> int:
    if not text: return 0
    try:
        enc = tok(text, return_tensors="pt")
        return int(enc["input_ids"].numel())
    except Exception:
        try: return int(len(tok.encode(text)))
        except Exception: return max(0, len(text)//2)

def build_virtual_slm_step_costs(base_parsed, base_cost, slm_model_name: str, slm_tokenizer, prompt_share_mode: str = "equal"):
    N = min(3, len(base_parsed))
    step_texts = [(base_parsed[i].get("chunk") or "") for i in range(N)]
    step_comp = [_count_tokens_tok(slm_tokenizer, txt) for txt in step_texts]
    tot_prompt = int(base_cost.get("prompt_tokens", 0))
    tot_comp   = int(base_cost.get("completion_tokens", 0))
    rem = max(0, tot_comp - sum(step_comp))
    if rem > 0 and N > 0: step_comp[-1] += rem
    if N <= 0: return []
    if prompt_share_mode == "proportional" and sum(step_comp) > 0:
        wsum = float(sum(step_comp))
        step_prompt = [int(round(tot_prompt * (c/wsum))) for c in step_comp]
        step_prompt[-1] += (tot_prompt - sum(step_prompt))
    else:
        base, r = divmod(tot_prompt, N)
        step_prompt = [base]*N; step_prompt[-1] += r
    out=[]
    for i in range(N):
        c = {"prompt_tokens": step_prompt[i], "completion_tokens": step_comp[i], "latency_ms": 0.0}
        apply_api_cost(c, slm_model_name)
        out.append(c)
    return out

def run_singlepass_chain(slm:HFChatCaller, problem:str, steps:List[str]):
    msgs = prompt_chain_raw(problem, steps)
    text, cost = slm.chat(msgs, max_new_tokens=1000, temperature=0.0)
    final = extract_final_line(text) or (extract_choice_letter(text) if is_csqa() else last_number(text)) or ""
    if not final:
        forced, cost2 = slm.chat(prompt_force_final_only(problem, text), max_new_tokens=16, temperature=0.0)
        f2 = extract_final_line(forced) or (extract_choice_letter(forced) if is_csqa() else None)
        if f2: final = f2
        for k in cost: cost[k] += cost2.get(k, 0.0)
    chunks = split_raw_chain_into_chunks(text, n=3)
    parsed=[]
    for ch in chunks:
        has, isok, val = extract_ans_from_chunk(ch, strict=False)
        parsed.append({"chunk": ch, "has": has, "isnum": isok, "value": val})
    return parsed, str(final), text, cost

# =========================
# step actions
# =========================
def run_detract_on_chunk(problem, subproblem, current_chunk, caller: HFChatCaller):
    msgs=prompt_edit_raw_chunk_minimal(problem, subproblem, current_chunk)
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

def run_escalate_to_final(problem, steps, executed_chunks, caller: HFChatCaller):
    text, cost = caller.chat(prompt_llm_only(problem), max_new_tokens=400, temperature=0.0)
    final = extract_final_line(text) or (extract_choice_letter(text) if is_csqa() else last_number(text)) or ""
    if not final:
        text2, cost2 = caller.chat(prompt_force_final_only(problem, text), max_new_tokens=16, temperature=0.0)
        f2 = extract_final_line(text2) or (extract_choice_letter(text2) if is_csqa() else None)
        for k in cost: cost[k]+=cost2.get(k,0.0)
        if f2: final=f2
    return (final if final else None), cost

# =========================
# router policy
# =========================
def run_policy(problem:str, steps:List[str], base_parsed:List[Dict[str,Any]], base_final:str,
               args, *, router=None, model_names=None, device=None,
               callers:Dict[str,'HFChatCaller']=None, slm:'HFChatCaller'=None,
               slm_virtual_step_costs: Optional[List[Dict[str,float]]] = None):

    N=len(steps)
    executed_chunks=[(p.get("chunk") if p else "") for p in (base_parsed or [])]
    chosen_actions=[]; chosen_models=[]
    final_answer=None; per_step_costs=[]
    total_tokens=0; total_api_cost=0.0
    modified_any=False

    detract_pool=[m.strip() for m in args.detract_models.split(",") if m.strip()]
    escalate_pool=[m.strip() for m in args.escalate_models.split(",") if m.strip()]

    def get_caller(name: str) -> 'HFChatCaller':
        if name not in callers: callers[name]=HFChatCaller(name, dtype=resolve_dtype(name))
        return callers[name]

    for t in range(N):
        units=["[CTX] "+problem]
        for k in range(t):
            units.append("[Q] "+steps[k]+"\n[A_SLM] "+(executed_chunks[k] or ""))
        q_text="[Q] "+steps[t]
        units.append(q_text+"\n[A_SLM] "+(executed_chunks[t] or ""))

        slm_sig=slm_sig_from_text(executed_chunks[t] or "")
        action, model = predict_action_and_model(router, model_names, units, q_text, slm_sig, t, device)

        # if SLM chosen for D/E, fallback to first non-SLM in respective pool
        if model=="SLM":
            pool=detract_pool if action=="Detract" else escalate_pool
            model=next((m for m in pool if m != args.slm_model), pool[0] if pool else args.slm_model)

        chosen_actions.append(action); chosen_models.append(model)
        slm_part = (slm_virtual_step_costs[t] if (slm_virtual_step_costs and t < len(slm_virtual_step_costs)) else {"prompt_tokens":0,"completion_tokens":0,"api_cost":0.0})

        if action=="Continue":
            total_tokens += slm_part["prompt_tokens"] + slm_part["completion_tokens"]
            total_api_cost += slm_part["api_cost"]
            per_step_costs.append({"step":t+1,"action":"Continue","model":"SLM","cost":slm_part})
            continue

        if action=="Detract":
            caller=get_caller(model)
            new_chunk, has, isok, val, d_cost = run_detract_on_chunk(problem, steps[t], executed_chunks[t], caller)
            executed_chunks[t]=new_chunk
            total_tokens += slm_part["prompt_tokens"]+slm_part["completion_tokens"] + d_cost["prompt_tokens"]+d_cost["completion_tokens"]
            total_api_cost += slm_part["api_cost"] + d_cost["api_cost"]
            modified_any=True
            per_step_costs.append({"step":t+1,"action":"Detract","model":model,
                                   "cost":{"prompt_tokens": slm_part["prompt_tokens"]+d_cost["prompt_tokens"],
                                           "completion_tokens": slm_part["completion_tokens"]+d_cost["completion_tokens"],
                                           "api_cost": slm_part["api_cost"]+d_cost["api_cost"]}})
            # if still no valid answer in chunk, escalate immediately
            if not isok:
                emodel = (escalate_pool[0] if escalate_pool else model)
                caller2=get_caller(emodel)
                esc_final2, esc_cost2 = run_escalate_to_final(problem, steps, executed_chunks, caller2)
                total_tokens += esc_cost2["prompt_tokens"] + esc_cost2["completion_tokens"]
                total_api_cost += esc_cost2["api_cost"]
                per_step_costs.append({"step":t+1,"action":"Escalate","model":emodel,"cost":esc_cost2})
                chosen_actions.append("Escalate"); chosen_models.append(emodel)
                if esc_final2: final_answer = esc_final2
                break

        if action=="Escalate":
            caller=get_caller(model)
            esc_final, esc_cost = run_escalate_to_final(problem, steps, executed_chunks, caller)
            total_tokens += slm_part["prompt_tokens"]+slm_part["completion_tokens"] + esc_cost["prompt_tokens"]+esc_cost["completion_tokens"]
            total_api_cost += slm_part["api_cost"] + esc_cost["api_cost"]
            per_step_costs.append({"step":t+1,"action":"Escalate","model":model,
                                   "cost":{"prompt_tokens": slm_part["prompt_tokens"]+esc_cost["prompt_tokens"],
                                           "completion_tokens": slm_part["completion_tokens"]+esc_cost["completion_tokens"],
                                           "api_cost": slm_part["api_cost"]+esc_cost["api_cost"]}})
            if esc_final: final_answer = esc_final
            break

    # (if we edited something) finalize once using an LLM
    if (not final_answer) and modified_any:
        finalizer = (escalate_pool[0] if escalate_pool else (detract_pool[0] if detract_pool else args.slm_model))
        caller = callers.get(finalizer) or HFChatCaller(finalizer, dtype=resolve_dtype(finalizer))
        callers[finalizer] = caller
        text, fcost = caller.chat(prompt_llm_only(problem), max_new_tokens=80, temperature=0.0)
        fin = extract_final_line(text) or (extract_choice_letter(text) if is_csqa() else last_number(text))
        if fin: final_answer=str(fin)
        total_tokens += fcost.get("prompt_tokens",0)+fcost.get("completion_tokens",0)
        total_api_cost += fcost.get("api_cost",0.0)

    if not final_answer:
        final_answer = base_final

    return {"actions":chosen_actions,"models":chosen_models,"executed_steps":executed_chunks,
            "final":final_answer,"correct":False,
            "total_api_cost":total_api_cost,"total_tokens":total_tokens}

# =========================
# main
# =========================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k","csqa","openbookqa","race","race-middle","race-high"], default="gsm8k")
    ap.add_argument("--split", default=None)
    ap.add_argument("--router_ckpt", default="ckpts/router_rl.pt")
    ap.add_argument("--slm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--llm_only_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--detract_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--escalate_models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--bank", default="bank_ret.jsonl")
    ap.add_argument("--ret_topk", type=int, default=0)       # disabled by default in clean build
    ap.add_argument("--ret_maxchars", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="predictions_router_clean.jsonl")
    ap.add_argument("--free_models", type=str, default="")
    ap.add_argument("--fp32_models", type=str, default="")
    ap.add_argument("--llm_only_mode", choices=["chain"], default="chain")
    ap.add_argument("--charge_slm_to_router", action="store_true", default=True)

    args=ap.parse_args()

    set_task(args.task)
    if args.free_models:
        for m in args.free_models.split(","):
            m=m.strip()
            if m: FREE_MODELS.add(m.lower())
    register_fp32_models(args.fp32_models)

    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router, model_names = load_router(args.router_ckpt, device)

    callers: Dict[str, HFChatCaller] = {}
    def get_caller(name: str, *, dtype: Optional[str]=None, profile: str="default") -> HFChatCaller:
        if name not in callers:
            callers[name]=HFChatCaller(name, dtype=(dtype or resolve_dtype(name)), profile=profile)
        return callers[name]

    # SLM (always) — use profile='slm'
    slm=get_caller(args.slm_model, dtype="float32", profile="slm")

    # (optional) LLM-only models for baseline
    llm_models=[s.strip() for s in args.llm_only_models.split(",") if s.strip()]
    for m in llm_models: get_caller(m, dtype=resolve_dtype(m), profile="slm")

    bank=load_case_bank_jsonl(args.bank); bm25=build_bm25_from_bank(bank) if bank else None  # kept; disabled by default via ret_topk=0

    # dataset loading
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
        ds_split = args.split or "test"
        ds_all = load_dataset("race", cfg)[ds_split]

    start=max(0,args.start)
    end=len(ds_all) if args.limit<=0 else min(len(ds_all), start+args.limit)
    if args.out == "predictions_router_clean.jsonl":
        args.out = f"predictions_{args.task}_router_clean.jsonl"

    methods = ["baseline", "router"] + [f"llm_only::{m}" for m in llm_models]
    method_stats={m:{"correct":0,"total":0,"total_tokens":0,"total_api_cost":0.0} for m in methods}

    dump=open(args.out,"w",encoding="utf-8")

    for i in tqdm(range(start,end), desc=f"Router inference ({args.task}/{ds_split})"):
        ds = ds_all[i]
        if args.task == "gsm8k":
            q = ds["question"].strip()
            gold_full = ds["answer"]
            gold = last_number(gold_full) or gold_full.strip()
            problem_text = q
        elif args.task == "csqa":
            stem = ds["question"].strip()
            labels = ds["choices"]["label"]; texts  = ds["choices"]["text"]
            options = "\n".join([f"{lab}. {txt}" for lab, txt in zip(labels, texts)])
            gold = ds["answerKey"].strip().upper()[:1]
            maxlab = (sorted(set(labels)) or ["E"])[-1]
            problem_text = f"{stem}\n\nOptions:\n{options}\n\nAnswer with the option letter (A-{maxlab})."
        elif args.task == "openbookqa":
            stem = ds["question_stem"].strip()
            labels = ds["choices"]["label"]; texts  = ds["choices"]["text"]
            options = "\n".join([f"{lab}. {txt}" for lab, txt in zip(labels, texts)])
            gold = ds["answerKey"].strip().upper()[:1]
            maxlab = (sorted(set(labels)) or ["D"])[-1]
            problem_text = f"{stem}\n\nOptions:\n{options}\n\nAnswer with the option letter (A-{maxlab})."
        elif args.task in ("race","race-middle","race-high"):
            passage = ds["article"].strip(); stem = ds["question"].strip()
            opts    = ds["options"]; labels  = ["A","B","C","D"][:len(opts)]
            options = "\n".join([f"{lab}. {txt}" for lab, txt in zip(labels, opts)])
            gold    = ds["answer"].strip().upper()[:1]
            problem_text = f"Passage:\n{passage}\n\nQuestion:\n{stem}\n\nOptions:\n{options}\n\nAnswer with the option letter (A-{labels[-1]})."

        # 1) Decompose (SLM)
        deco_text, deco_cost = slm.chat(prompt_decompose_compact(problem_text,"unknown"), max_new_tokens=160, temperature=0.0)
        deco_tokens = deco_cost["prompt_tokens"] + deco_cost["completion_tokens"]
        deco_api    = deco_cost["api_cost"]

        # parse 3 steps
        steps=[]
        m=re.search(r'do\s*:\s*', deco_text, flags=re.I); sub=deco_text[m.end():] if m else deco_text
        for raw in (sub or "").splitlines():
            ln=re.sub(r'^\s*(?:\(?\d+\)?\s*[:.)-]\s*|step\s*\d+\s*[:.)-]\s*|[-*•]\s*)','',raw.strip(), flags=re.I)
            if 2<=len(ln.split())<=14: steps.append(ln.strip())
            if len(steps)>=3: break
        if len(steps)<3: steps=(steps+["Choose the most plausible option."]*3)[:3]

        # 2) Baseline (SLM)
        base_parsed, base_final, base_text, base_cost = run_singlepass_chain(slm, problem_text, steps)
        base_tokens = base_cost["prompt_tokens"] + base_cost["completion_tokens"]
        base_api    = base_cost["api_cost"]
        base_ok = judge_answer(gold, base_final or "")

        method_stats["baseline"]["total"] += 1
        if base_ok: method_stats["baseline"]["correct"] += 1
        method_stats["baseline"]["total_tokens"] += (deco_tokens + base_tokens)
        method_stats["baseline"]["total_api_cost"] += (deco_api + base_api)

        # 3) SLM virtual step costs
        slm_virtual_step_costs = build_virtual_slm_step_costs(base_parsed, base_cost, args.slm_model, slm.tokenizer, prompt_share_mode="equal")

        # 4) Router
        router_out = run_policy(
            problem_text, steps, base_parsed, base_final, args,
            router=router, model_names=model_names, device=device,
            callers=callers, slm=slm, slm_virtual_step_costs=slm_virtual_step_costs
        )
        if args.charge_slm_to_router:
            router_out["total_tokens"]  += deco_tokens
            router_out["total_api_cost"]+= deco_api
        router_ok = judge_answer(gold, router_out["final"] or "")
        router_out["correct"] = bool(router_ok)
        st=method_stats["router"]; st["total"]+=1
        if router_ok: st["correct"]+=1
        st["total_tokens"]+=router_out["total_tokens"]; st["total_api_cost"]+=router_out["total_api_cost"]

        # 5) LLM-only (chain)
        llm_only_results={}
        for mname in llm_models:
            msgs = prompt_chain_raw(problem_text, steps)
            final_text, cost = callers[mname].chat(msgs, max_new_tokens=900, temperature=0.0)
            final = extract_final_line(final_text) or (extract_choice_letter(final_text) if is_csqa() else last_number(final_text)) or ""
            ok = judge_answer(gold, final or "")
            llm_only_results[mname] = {"final": final, "correct": bool(ok)}
            tag=f"llm_only::{mname}"
            st = method_stats[tag] = method_stats.get(tag, {"correct":0,"total":0,"total_tokens":0,"total_api_cost":0.0})
            st["total"] += 1
            if ok: st["correct"] += 1
            st["total_tokens"] += cost["prompt_tokens"] + cost["completion_tokens"]
            st["total_api_cost"] += cost["api_cost"]

        # 6) dump per-problem
        rec={"id": f"{args.task}-{ds_split}-{i:05d}",
             "problem": problem_text, "steps": steps,
             "baseline":{"final": base_final,"correct": bool(base_ok),"raw_chain": base_text,"parsed": base_parsed,
                         "cost":{"decompose": deco_cost,"chain": base_cost,
                                 "total_tokens": (deco_tokens + base_tokens),
                                 "total_api_cost": (deco_api + base_api)}},
             "router": router_out, "llm_only": llm_only_results, "gold": str(gold)}
        dump.write(json.dumps(rec, ensure_ascii=False)+"\n")

    dump.close()

    print("\n=== Summary (clean) ===")
    for m,st in method_stats.items():
        acc=(st["correct"]/st["total"]) if st["total"]>0 else 0.0
        print(f"[{m}] Acc: {acc:.4f} ({st['correct']}/{st['total']}) | Tokens={st['total_tokens']} | $={st['total_api_cost']:.6f}")

    print(f"\nPer-problem dump -> {args.out}")

    for c in list(callers.values()):
        try: c.close()
        except: pass

if __name__ == "__main__":
    main()
