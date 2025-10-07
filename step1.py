#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, math, random, argparse, copy, types, csv, datetime, zlib
os.environ["CUDA_VISIBLE_DEVICES"]=os.environ.get("CUDA_VISIBLE_DEVICES","2")
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# ================= Basics =================
ACTIONS = ["Continue", "Detract", "Escalate"]
SEED_DEFAULT = 42

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def split_teacher_label(s: str) -> Optional[Tuple[str, str]]:
    if not s or "@" not in s: return None
    a, m = s.split("@", 1); a=a.strip(); m=m.strip()
    if (not a) or (not m): return None
    return a, m

def safe_get(d, k, default=None):
    v = d.get(k) if isinstance(d, dict) else None
    return default if v is None else v

def forbid_slm_on_de(mdl_logits: torch.Tensor,
                     act_ids: torch.Tensor,
                     slm_idx: int = 0,
                     neg_inf: float = None):
    """ D/E 예측에서 SLM(인덱스 0) 선택 못하게 큰 음수로 마스킹 """
    out = mdl_logits.clone()
    if out.numel() == 0:
        return out
    mask = (act_ids != 0)
    if not mask.any():
        return out
    if neg_inf is None:
        neg_inf = -1e4 if out.dtype in (torch.float16, torch.half) else -1e9
    neg_val = out.new_tensor(neg_inf)
    out[mask, slm_idx] = neg_val
    return out

# ================= Data types =================
@dataclass
class StepItem:
    idx: int
    subq: str
    context: str
    actions_blob: Dict[str, Any]
    teacher_label: str

@dataclass
class Episode:
    id: str
    task: str
    problem_text: str
    steps: List[StepItem]
    solution: Optional[str]
    sg_step_idx: Optional[int] = None
    sg_action: Optional[str] = None
    question_ctx: Optional[str] = None

@dataclass
class TrainSample:
    unit_texts: List[str]
    q_text: str
    slm_sig: np.ndarray
    step_idx: int
    gold_action: int
    gold_model: int
    valid_actions: np.ndarray
    valid_models_for_gold: np.ndarray
    valid_models_by_action: np.ndarray
    split: str

# ================= IO & parsing =================
def read_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    out=[]
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except: pass
    return out

def _mcqa_ctx_from_query(qobj: dict) -> str:
    q = normalize(qobj.get("question") or qobj.get("problem") or "")
    chs = qobj.get("choices") or []
    if isinstance(chs, list) and chs and isinstance(chs[0], dict) and "label" in chs[0]:
        lines = [f"{c.get('label')}. {normalize(c.get('text') or '')}" for c in chs]
        return (q + "\n" + "\n".join(lines)).strip()
    return q

def _resolve_sample_gold(obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    cand = None
    for k in ["sample_gold", "sampleGold", "samplegold"]:
        if isinstance(obj.get(k), dict):
            cand = obj[k]; break
    if not isinstance(cand, dict): return None, None
    idx = cand.get("step_idx")
    act = cand.get("action")
    if not isinstance(act, str) or "@" not in act: return None, None
    if not isinstance(idx, int):
        a_name, _ = split_teacher_label(act)
        if a_name == "Continue": idx = 0
        else: return None, None
    return int(idx), act.strip()

def parse_episodes(raw_objs: List[Dict[str, Any]], allowed_models: List[str]) -> List[Episode]:
    def map_model(name: str) -> str:
        n = (name or "").strip()
        return "SLM" if n in ["Qwen/Qwen2.5-1.5B-Instruct", "SLM"] else n

    eps=[]
    for obj in raw_objs:
        q = obj.get("query", {}) or {}
        task = (obj.get("task") or "").lower()
        qctx = _mcqa_ctx_from_query(q) if task in ["mcqa","csqa"] else ""
        problem = normalize(q.get("problem") or "") or qctx

        steps=[]
        for st in obj.get("steps") or []:
            pair = split_teacher_label(st.get("teacher_label"))
            if not pair: continue
            a, m = pair; m = map_model(m)
            if a not in ACTIONS or m not in allowed_models: continue
            steps.append(
                StepItem(
                    idx=int(st.get("idx", 0)),
                    subq=normalize(st.get("subquestion") or ""),
                    context=normalize(st.get("context") or ""),
                    actions_blob=st.get("actions") or {},
                    teacher_label=f"{a}@{m}"
                )
            )

        sg_idx, sg_act = _resolve_sample_gold(obj)
        if isinstance(sg_act, str) and "@" in sg_act:
            a_tmp, m_tmp = split_teacher_label(sg_act)
            sg_act = f"{a_tmp}@{map_model(m_tmp)}" if a_tmp in ACTIONS else None

        if steps:
            eps.append(Episode(
                id=obj.get("id") or "",
                task=obj.get("task") or "",
                problem_text=problem,
                steps=steps,
                solution=q.get("solution"),
                sg_step_idx=(sg_idx if isinstance(sg_idx, int) else None),
                sg_action=sg_act if isinstance(sg_act, str) else None,
                question_ctx=(qctx or None)
            ))
    return eps

# ================= Step extraction =================
def extract_action_text(step: StepItem, key: str) -> str:
    blob = step.actions_blob.get(key) or {}
    txt = (safe_get(blob, "step_output_text", "") or safe_get(blob, "step_answer_only", "") or "").strip()
    cut = txt.find("FINAL:")
    if cut != -1: txt = txt[:cut].strip()
    return txt

def extract_slm_signals(step: StepItem, slm_key="Continue@SLM") -> np.ndarray:
    """ 간단한 SLM 품질 신호(5D) """
    blob = step.actions_blob.get(slm_key) or {}
    format_ok = 1.0 if bool(blob.get("format_ok")) else 0.0
    ans_is_numeric = 1.0 if bool(blob.get("ans_is_numeric")) else 0.0
    txt = (blob.get("step_output_text") or blob.get("step_answer_only") or "") or ""
    s = txt.replace("[", "").replace("]", " ")
    chars=[c for c in s]; L=max(1,len(chars))
    digit_ratio = sum(c.isdigit() for c in chars)/L
    eq_ratio = sum(c in "+-*/=^" for c in chars)/L
    token_len = float(len(s.split()))
    return np.array([format_ok, ans_is_numeric, digit_ratio, eq_ratio, token_len], dtype=np.float32)

# ================= Acceptable sets =================
def build_valid_action_mask(step: StepItem) -> np.ndarray:
    mask = np.zeros(3, dtype=np.float32)
    for key,blob in (step.actions_blob or {}).items():
        if not isinstance(blob, dict): continue
        if blob.get("final_correct_if_applied") is True:
            pair = split_teacher_label(key)
            if not pair: continue
            a,_ = pair
            if a in ACTIONS: mask[ACTIONS.index(a)] = 1.0
    if mask.sum()==0:
        a,_ = split_teacher_label(step.teacher_label)
        mask[ACTIONS.index(a)] = 1.0
    return mask

def build_valid_model_mask_for_action(step: StepItem, allowed_models: List[str], action_idx: int) -> np.ndarray:
    K=len(allowed_models); mask=np.zeros(K, dtype=np.float32)
    a_name = ACTIONS[action_idx]
    for key,blob in (step.actions_blob or {}).items():
        pair = split_teacher_label(key)
        if not pair: continue
        a,m = pair
        if a!=a_name: continue
        ok = (blob.get("final_correct_if_applied") is True)
        if ok and m in allowed_models:
            mask[allowed_models.index(m)] = 1.0
    return mask

def build_valid_model_mask_all(step: StepItem, allowed_models: List[str]) -> np.ndarray:
    K=len(allowed_models)
    out = np.zeros((3,K), dtype=np.float32)
    for a_idx in range(3):
        out[a_idx,:] = build_valid_model_mask_for_action(step, allowed_models, a_idx)
    return out

# ================= Build samples =================
def build_train_samples(eps: List[Episode], allowed_models: List[str],
                        split_ratios=(0.7,0.15,0.15), seed=42,
                        label_source: str = "auto"):
    """
    label_source:
      - "auto": sample_gold 있으면 사용, 없으면 teacher 사용
      - "sample_gold": sample_gold만 사용(없으면 teacher 폴백)
      - "teacher": teacher만 사용
    """
    rng=random.Random(seed); rng.shuffle(eps)
    N=len(eps)
    n_tr=int(N*split_ratios[0]); n_dv=int(N*split_ratios[1]); n_te=N-n_tr-n_dv
    marks=(["train"]*n_tr)+(["dev"]*n_dv)+(["test"]*n_te)
    model2idx={m:i for i,m in enumerate(allowed_models)}

    sg_present = 0; sg_used = 0; sg_model_miss = 0; sg_idx_miss = 0

    def pick_steps(ep: Episode) -> List[StepItem]:
        nonlocal sg_used, sg_idx_miss
        use_sg = (label_source == "sample_gold") or (label_source == "auto" and ep.sg_action and ep.sg_step_idx is not None)
        if use_sg:
            target = None
            for st in ep.steps:
                if st.idx == ep.sg_step_idx:
                    target = st; break
            if target is None:
                sg_idx_miss += 1
                return ep.steps
            sg_used += 1
            return [target]
        else:
            return ep.steps

    for ep in eps:
        if ep.sg_action is not None and ep.sg_step_idx is not None:
            sg_present += 1

    tr=[]; dv=[]; te=[]
    for ep,sp in zip(eps, marks):
        ctx = ep.problem_text or ep.question_ctx or (ep.steps[0].context if ep.steps else "")
        chosen_steps = pick_steps(ep)
        for step in chosen_steps:
            units=[]
            if ctx: units.append("[CTX] " + ctx)
            prev_hist = [s for s in ep.steps if s.idx < step.idx]
            for prev in prev_hist:
                qa = "[Q] " + (prev.subq or "(no-subq)")
                gold_text = extract_action_text(prev, prev.teacher_label)
                units.append(qa + "\n[A] " + (gold_text or ""))

            q_text = "[Q] " + (step.subq or "(no-subq)")
            slm_text = extract_action_text(step, "Continue@SLM")
            units.append(q_text + ("\n[A_SLM] " + slm_text if slm_text else ""))

            if (label_source in ["sample_gold","auto"]) and (ep.sg_action and ep.sg_step_idx == step.idx):
                a_str, m_str = split_teacher_label(ep.sg_action)
                if m_str not in allowed_models:
                    sg_model_miss += 1
                    a_str, m_str = split_teacher_label(step.teacher_label)
            else:
                a_str, m_str = split_teacher_label(step.teacher_label)

            ga = ACTIONS.index(a_str)
            gm = model2idx[m_str]

            slm_sig = extract_slm_signals(step, "Continue@SLM")
            vact = build_valid_action_mask(step)
            vmdl_gold = build_valid_model_mask_for_action(step, allowed_models, ga)
            vmdl_all = build_valid_model_mask_all(step, allowed_models)

            sample = TrainSample(units, q_text, slm_sig, step.idx, ga, gm, vact, vmdl_gold, vmdl_all, sp)
            (tr if sp=="train" else dv if sp=="dev" else te).append(sample)

            # teacher 모드에서만 Escalate로 에피소드 중단(과거 호환)
            if (label_source == "teacher") and (a_str=="Escalate"):
                break

    def dist(name, L):
        c={"Continue":0,"Detract":0,"Escalate":0}
        for s in L: c[ACTIONS[s.gold_action]]+=1
        print(f"[{name}] gold action distribution: {c}")

    print(f"Data sizes (flattened): train={len(tr)} dev={len(dv)} test={len(te)}")
    print(f"[label_source] mode = {label_source}")
    dist("train",tr); dist("dev",dv); dist("test",te)
    if sg_present:
        print(f"[SG] episodes with sample_gold present: {sg_present}/{len(eps)}")
    if label_source in ["auto","sample_gold"]:
        print(f"[SG] used sample_gold episodes: {sg_used} (idx_miss={sg_idx_miss}, model_not_allowed={sg_model_miss})")
    return tr,dv,te

# ================= AUX features (6D) =================
UNITS_RE = re.compile(r"\b(cm|mm|km|kg|g|miles?|hours?|mins?|seconds?|percent|%)\b", re.I)
NUM_RE   = re.compile(r"\b\d+(\.\d+)?\b")
EQ_KWS = ["equation","system","integral","derivative"]
REASON_KWS = ["prove","show","explain","why"]

def aux_vec_from_qtext(q_text: str) -> np.ndarray:
    """ 6-D AUX: [numbers_ratio, units_flag, ops_ratio, avg_token_len, has_eq_kw, has_reason_kw] """
    s = (q_text or "").replace("[Q]", "").strip().lower()
    tokens = s.split(); T = max(1, len(tokens))
    chars = list(s); L = max(1, len(chars))
    numbers_ratio = float(len(NUM_RE.findall(s)))/T
    units_flag    = 1.0 if UNITS_RE.search(s) else 0.0
    ops_ratio     = sum(c in "+-*/=^" for c in chars)/L
    avg_toklen    = (np.mean([len(t) for t in tokens]) if tokens else 0.0)
    has_eq_kw     = 1.0 if any(k in s for k in EQ_KWS) else 0.0
    has_reason_kw = 1.0 if any(k in s for k in REASON_KWS) else 0.0
    return np.array([numbers_ratio, units_flag, ops_ratio, avg_toklen, has_eq_kw, has_reason_kw], dtype=np.float32)

def aux_dim_reduced() -> int: return 6
def batch_aux_features(q_texts: List[str]) -> torch.Tensor:
    arr = np.stack([aux_vec_from_qtext(t) for t in q_texts], axis=0)
    return torch.from_numpy(arr)

# ================= Encoders =================
class MeanPooler(nn.Module):
    def forward(self, last_hidden, attn_mask):
        m = attn_mask.unsqueeze(-1).float()
        return (last_hidden*m).sum(1) / m.sum(1).clamp(min=1e-6)

class BertEncoder(nn.Module):
    def __init__(self, name="bert-base-uncased", max_len=384, trainable=True, local_files_only=False):
        super().__init__()
        self.max_len = max_len
        self.pool = MeanPooler()
        self._cache = OrderedDict(); self._cache_cap = 50000
        try:
            self.tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)
        except Exception as e:
            print(f"[HF-OFFLINE] tokenizer fallback: {e}")
            self.tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)
        try:
            self.bert= AutoModel.from_pretrained(name, local_files_only=local_files_only)
        except Exception as e:
            print(f"[HF-OFFLINE] model fallback: {e}")
            self.bert= AutoModel.from_pretrained(name, local_files_only=True)
        if not trainable:
            for p in self.bert.parameters(): p.requires_grad=False

    def _cache_put(self, key: str, vec: torch.Tensor):
        self._cache[key] = vec; self._cache.move_to_end(key)
        if len(self._cache) > self._cache_cap: self._cache.popitem(last=False)

    def encode_texts(self, texts: List[str], device):
        result: List[Optional[torch.Tensor]] = [None]*len(texts)
        miss_indices, miss_texts = [], []
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is not None:
                self._cache.move_to_end(t)
                result[i] = v
            else:
                miss_indices.append(i); miss_texts.append(t)

        if miss_texts:
            enc = self.tok(miss_texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            out = self.bert(**enc)
            pooled = self.pool(out.last_hidden_state, enc["attention_mask"]).detach()  # [B,768]
            pooled_cpu = pooled.to("cpu", dtype=torch.float16)
            for idx, vec in zip(miss_indices, pooled_cpu):
                self._cache_put(texts[idx], vec)
                result[idx] = vec
        Z = torch.stack([r for r in result], dim=0).to(device=device, dtype=torch.float32)
        return Z  # [B,768]

# ================= Router =================
class GRURouter(nn.Module):
    def __init__(self, num_models, aux_dim, slm_sig_dim,
                 bert_name="bert-base-uncased", max_len=384,
                 freeze_bert=False, gru_hidden=384, step_emb_dim=16, proj_dim=512,
                 p_drop=0.1, bert_local_files_only=False):
        super().__init__()
        self.enc = BertEncoder(bert_name, max_len=max_len, trainable=not freeze_bert,
                               local_files_only=bert_local_files_only)
        self.gru = nn.GRU(768, gru_hidden, batch_first=True)
        self.step_emb = nn.Embedding(64, step_emb_dim)

        self.model_vecs = nn.Parameter(torch.randn(num_models, 768)*0.02)
        self.sim_temp   = nn.Parameter(torch.tensor(0.7))

        self._aux_dim = int(aux_dim)
        self._slm_dim = int(slm_sig_dim)
        cur_in = 768 + 768 + 3 + self._aux_dim + self._slm_dim + step_emb_dim
        self.cur_mlp = nn.Sequential(
            nn.Linear(cur_in, proj_dim), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(proj_dim, proj_dim), nn.ReLU()
        )
        self.fuse = nn.Sequential(nn.Linear(gru_hidden + proj_dim, proj_dim), nn.ReLU())
        self.dropout = nn.Dropout(p_drop)
        self.ln_fuse = nn.LayerNorm(proj_dim)
        self.act_head = nn.Linear(proj_dim, 3)
        self.mdl_head = nn.Linear(proj_dim, num_models)

        self.register_buffer("aux_mean", torch.zeros(self._aux_dim))
        self.register_buffer("aux_std", torch.ones(self._aux_dim))
        self.register_buffer("slm_mean", torch.zeros(self._slm_dim))
        self.register_buffer("slm_std", torch.ones(self._slm_dim))

    def set_feature_stats(self, aux_mean, aux_std, slm_mean, slm_std):
        with torch.no_grad():
            if self._aux_dim == 0:
                self.aux_mean = torch.zeros(0, device=self.act_head.weight.device)
                self.aux_std  = torch.ones(0, device=self.act_head.weight.device)
            else:
                self.aux_mean.copy_(aux_mean.clone().detach())
                self.aux_std.copy_(torch.clamp(aux_std.clone().detach(), min=1e-6))
            if self._slm_dim == 0:
                self.slm_mean = torch.zeros(0, device=self.act_head.weight.device)
                self.slm_std  = torch.ones(0, device=self.act_head.weight.device)
            else:
                self.slm_mean.copy_(slm_mean.clone().detach())
                self.slm_std.copy_(torch.clamp(slm_std.clone().detach(), min=1e-6))

    def encode_units(self, batch_units: List[List[str]], device):
        flat=[]; offsets=[0]
        for units in batch_units:
            flat += units; offsets.append(offsets[-1]+len(units))
        Z = self.enc.encode_texts(flat, device=device)
        lengths=[]; chunks=[]
        for i in range(len(batch_units)):
            a=offsets[i]; b=offsets[i+1]
            chunks.append(Z[a:b]); lengths.append(b-a)
        Tmax=max(lengths); B=len(batch_units)
        X=torch.zeros(B, Tmax, 768, device=Z.device)
        for i,ch in enumerate(chunks):
            X[i,:ch.size(0),:] = ch
        return X, lengths

    def forward(self, batch_units: List[List[str]], q_texts: List[str], slm_sigs_in: torch.Tensor, step_idx: torch.Tensor):
        self.gru.flatten_parameters()
        device = step_idx.device
        X, lengths = self.encode_units(batch_units, device=device)
        packed = pack_padded_sequence(X, lengths=lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed); h_last = h.squeeze(0)

        B = X.size(0)
        q_emb = torch.stack([X[i, lengths[i]-1, :] for i in range(B)], dim=0)
        past_mean=[]
        for i in range(B):
            L=lengths[i]-1
            past_mean.append(torch.zeros(768, device=device) if L<=0 else X[i,:L,:].mean(0))
        past_mean = torch.stack(past_mean, dim=0)

        sims = torch.matmul(q_emb, self.model_vecs.t())
        prob = F.softmax(self.sim_temp * sims, dim=1).clamp(min=1e-6)
        sim_max = sims.max(1, keepdim=True).values
        sim_min = sims.min(1, keepdim=True).values
        sim_ent = -(prob * prob.log()).sum(1, keepdim=True)

        if self.aux_mean.numel() > 0:
            aux = batch_aux_features(q_texts).to(device).float()
            aux = (aux - self.aux_mean) / self.aux_std
        else:
            aux = torch.zeros((B,0), device=device)

        if self.slm_mean.numel() > 0:
            slm_sigs = (slm_sigs_in.float() - self.slm_mean) / self.slm_std
        else:
            slm_sigs = torch.zeros((B,0), device=device)

        stepv = self.step_emb(step_idx.clamp(min=0, max=self.step_emb.num_embeddings-1))
        cur_feat = torch.cat([q_emb, past_mean, sim_max, sim_min, sim_ent, aux, slm_sigs, stepv], dim=1)
        cur_h = self.cur_mlp(cur_feat)

        fused = self.fuse(torch.cat([h_last, cur_h], dim=1))
        fused = self.ln_fuse(self.dropout(fused))
        act_logits = self.act_head(fused)
        mdl_logits = self.mdl_head(fused)
        return act_logits, mdl_logits

# ================= Collate & iters =================
def collate(samples: List[TrainSample], device):
    units=[s.unit_texts for s in samples]
    qtexts=[s.q_text for s in samples]
    slm = torch.from_numpy(np.stack([s.slm_sig for s in samples], axis=0)).to(device)
    step=torch.tensor([s.step_idx for s in samples], dtype=torch.long, device=device)
    ga = torch.tensor([s.gold_action for s in samples], dtype=torch.long, device=device)
    gm = torch.tensor([s.gold_model for s in samples], dtype=torch.long, device=device)
    vact = torch.from_numpy(np.stack([s.valid_actions for s in samples], axis=0)).to(device).float()
    vmdl = torch.from_numpy(np.stack([s.valid_models_for_gold for s in samples], axis=0)).to(device).float()
    vmdl_all = torch.from_numpy(np.stack([s.valid_models_by_action for s in samples], axis=0)).to(device).float()
    return units,qtexts,slm,step,ga,gm,vact,vmdl,vmdl_all

def iter_weighted_batches(samples: List[TrainSample], per_c:int, per_d:int, per_e:int, iters:int):
    pools={0:[],1:[],2:[]}
    for i,s in enumerate(samples): pools[s.gold_action].append(i)
    keys=[0,1,2]; per={0:per_c,1:per_d,2:per_e}
    for _ in range(iters):
        cur=[]
        for k in keys:
            pool=pools[k]
            if not pool: continue
            need = per[k]
            cur += [ random.choice(pool) for __ in range(need) ]
        random.shuffle(cur)
        yield cur

def iter_eval_batches(N: int, batch_size: int):
    for i in range(0, N, batch_size):
        yield list(range(i, min(i+batch_size, N)))

# ================= Eval helpers =================
@torch.no_grad()
def evaluate(model, samples, model_names, device, batch_size=16,
             la_bias: Optional[torch.Tensor]=None):
    model.eval()
    N=len(samples)
    gold_a=[]; pred_a=[]; gold_m=[]; pred_m=[]
    for idxs in iter_eval_batches(N, batch_size):
        batch=[samples[i] for i in idxs]
        units,qtexts,slm,step,ga,gm,_,_,_=collate(batch, device)
        act_logits, mdl_logits = model(units,qtexts,slm,step)
        if la_bias is not None: act_logits = act_logits + la_bias.view(1,-1)
        pa = act_logits.argmax(1)
        blended = forbid_slm_on_de(mdl_logits, pa, slm_idx=0)
        pm = blended.argmax(1)
        pm = torch.where(pa==0, torch.zeros_like(pm), pm)
        gold_a += ga.cpu().tolist(); pred_a += pa.cpu().tolist()
        gold_m += gm.cpu().tolist(); pred_m += pm.cpu().tolist()

    M = int(model.mdl_head.out_features)
    names = list(model_names) if isinstance(model_names,(list,tuple)) and len(model_names)==M else [f"m{i}" for i in range(M)]
    mat_a = np.zeros((3,3), dtype=int)
    mat_m = np.zeros((M,M), dtype=int)
    for g,p in zip(gold_a,pred_a): mat_a[g,p] += 1
    for g,p in zip(gold_m,pred_m):
        if 0 <= g < M and 0 <= p < M: mat_m[g,p] += 1
    hist={0:0,1:0,2:0}
    for x in pred_a: hist[x]+=1
    act_acc = (np.array(gold_a)==np.array(pred_a)).mean() if N else 0.0
    mdl_acc = (np.array(gold_m)==np.array(pred_m)).mean() if N else 0.0
    joint   = np.mean([ (ga==pa) and (gm==pm) for (ga,gm),(pa,pm) in zip(zip(gold_a,gold_m), zip(pred_a,pred_m)) ]) if N else 0.0
    print(f"[pred action dist] C={hist[0]} D={hist[1]} E={hist[2]}")
    print("Confusion (ACTION gold×pred):")
    print(f"{'':15s}{'Continue':12s}{'Detract':12s}{'Escalate':12s}")
    for i,r in enumerate(["Continue","Detract","Escalate"]):
        print(f"{r:15s}{mat_a[i,0]:<12d}{mat_a[i,1]:<12d}{mat_a[i,2]:<12d}")
    head="".join([f"{n[:14]:14s}" for n in names])
    print("Confusion (MODEL gold×pred):")
    print(f"{'':15s}{head}")
    for i,n in enumerate(names):
        row="".join([f"{mat_m[i,j]:<14d}" for j in range(M)])
        print(f"{n[:14]:15s}{row}")
    # Macro-F1
    def f1_macro(gold, pred, num_classes=3):
        gold = np.array(gold); pred = np.array(pred)
        f1s=[]
        for c in range(num_classes):
            tp = np.sum((gold==c)&(pred==c))
            fp = np.sum((gold!=c)&(pred==c))
            fn = np.sum((gold==c)&(pred!=c))
            precision = tp/(tp+fp+1e-9)
            recall    = tp/(tp+fn+1e-9)
            f1 = 2*precision*recall/(precision+recall+1e-9)
            f1s.append(f1)
        return float(np.mean(f1s)), f1s
    f1_macro_all, f1_each = f1_macro(gold_a, pred_a, 3)
    print(f"[ACTION Macro-F1] macro={f1_macro_all:.4f} | per-class (C/D/E) = {np.round(f1_each,4)}")
    return act_acc, mdl_acc, joint

@torch.no_grad()
def compute_metrics(model, samples, model_names, device, batch_size=16,
                    la_bias: Optional[torch.Tensor]=None):
    model.eval()
    N=len(samples)
    gold_a=[]; gold_m=[]; pred_a=[]; pred_m=[]; soft_a=[]; soft_joint=[]; route_ok_list=[]
    for idxs in iter_eval_batches(N, batch_size):
        batch=[samples[i] for i in idxs]
        units,qtexts,slm,step,ga,gm,vact,_,vmdl_all = collate(batch, device)
        act_logits, mdl_logits = model(units,qtexts,slm,step)
        if la_bias is not None: act_logits = act_logits + la_bias.view(1,-1)
        pa = act_logits.argmax(1)
        blended = forbid_slm_on_de(mdl_logits, pa, slm_idx=0)
        pm = blended.argmax(1)
        pm = torch.where(pa==0, torch.zeros_like(pm), pm)
        rows = torch.arange(pa.size(0), device=pa.device)
        ok_a = vact[rows, pa].bool()
        ok_m = vmdl_all[rows, pa, pm].bool()
        ok_joint = ok_a & ( (pa==0) | ok_m )
        route_ok = vmdl_all[rows, pa, pm].bool()
        route_ok_list += route_ok.cpu().tolist()
        soft_a += ok_a.cpu().tolist()
        soft_joint += ok_joint.cpu().tolist()
        gold_a += ga.cpu().tolist(); gold_m += gm.cpu().tolist()
        pred_a += pa.cpu().tolist(); pred_m += pm.cpu().tolist()
    gold_a=np.array(gold_a); gold_m=np.array(gold_m)
    pred_a=np.array(pred_a); pred_m=np.array(pred_m)
    soft_a=np.array(soft_a); soft_joint=np.array(soft_joint)
    route_ok=np.array(route_ok_list)
    # macro-F1
    f1s=[]
    for c in range(3):
        tp=np.sum((gold_a==c)&(pred_a==c))
        fp=np.sum((gold_a!=c)&(pred_a==c))
        fn=np.sum((gold_a==c)&(pred_a!=c))
        precision=tp/(tp+fp+1e-9); recall=tp/(tp+fn+1e-9)
        f1=2*precision*recall/(precision+recall+1e-9)
        f1s.append(f1)
    return {
        "gold_action_acc": float((gold_a==pred_a).mean()) if N else 0.0,
        "gold_joint_acc": float(np.mean((gold_a==pred_a)&(gold_m==pred_m))) if N else 0.0,
        "soft_action_acc": float(soft_a.mean()) if N else 0.0,
        "soft_joint_acc": float(soft_joint.mean()) if N else 0.0,
        "route_ok": float(route_ok.mean()) if N else 0.0,
        "action_macro_f1": float(np.mean(f1s))
    }

# ================= Loss utils (공용) =================
def soft_ce_from_masks(logits, gold_idx, valid_mask, gold_weight=0.8):
    B,C = logits.size()
    tgt = torch.zeros_like(logits, dtype=torch.float)
    for i in range(B):
        g = gold_idx[i].item()
        vm = valid_mask[i].clone(); vm[g] = 0.0
        k = int(vm.sum().item())
        if k > 0:
            tgt[i,g] = gold_weight
            tgt[i, vm.bool()] = (1.0 - gold_weight) / k
        else:
            tgt[i,g] = 1.0
    logp = F.log_softmax(logits, dim=1)
    return -(tgt * logp).sum(1).mean()

def entropy_bonus(logits):
    p = F.softmax(logits, dim=1).clamp(min=1e-6)
    return (-(p * p.log()).sum(1)).mean()

def kl_penalty(new_logits, old_logits):
    p_new = F.softmax(new_logits, dim=1).clamp(min=1e-8)
    logp_new = p_new.log()
    logp_old = F.log_softmax(old_logits, dim=1)
    kl = (p_new * (logp_new - logp_old)).sum(1)
    return kl.mean()

# ================= Feature stats & init =================
def compute_feature_stats(samples: List[TrainSample], use_aux: bool, use_slm: bool):
    if use_aux:
        AUX=np.stack([aux_vec_from_qtext(s.q_text) for s in samples],0)
        aux_mean = torch.from_numpy(AUX.mean(0)).float()
        aux_std  = torch.from_numpy(AUX.std(0)+1e-6).float()
    else:
        aux_mean = torch.zeros(0); aux_std=torch.ones(0)
    if use_slm:
        SLM=np.stack([s.slm_sig for s in samples],0)
        slm_mean = torch.from_numpy(SLM.mean(0)).float()
        slm_std  = torch.from_numpy(SLM.std(0)+1e-6).float()
    else:
        slm_mean = torch.zeros(0); slm_std=torch.ones(0)
    print("[feature] standardized AUX/SLM signals with train-set stats.")
    return aux_mean, aux_std, slm_mean, slm_std

def init_biases_from_label_stats(model, tr_samples, model_names, device):
    counts=[0,0,0]; mdl_counts_nc=[0]*len(model_names)
    for s in tr_samples:
        counts[s.gold_action]+=1
        if s.gold_action!=0: mdl_counts_nc[s.gold_model]+=1
    tot=sum(counts)+1e-9
    priors=[c/tot for c in counts]
    logit_bias=[math.log(max(1e-5,p)/(1-p)) for p in priors]
    with torch.no_grad():
        model.act_head.bias.copy_(torch.tensor(logit_bias, device=device))
    print(f"[init] action priors={np.round(priors,3)} -> act_head.bias={np.round(logit_bias,3)}")

    tot_nc=sum(mdl_counts_nc)+1e-9
    if tot_nc>0:
        pri_m=[c/tot_nc for c in mdl_counts_nc]
        logit_b=[math.log(max(1e-5,p)/(1-p)) for p in pri_m]
        with torch.no_grad():
            model.mdl_head.bias.copy_(torch.tensor(logit_b, device=device))
        print(f"[init] model priors(non-Continue)={np.round(pri_m,3)} -> mdl_head.bias={np.round(logit_b,3)}")

@torch.no_grad()
def initialize_from_data(model, samples: List[TrainSample], device, max_batches=300, batch_size=24):
    """ non-Continue에서 모델별 평균 q_emb로 model_vecs 초기화 """
    model.eval()
    K = model.model_vecs.size(0)
    sums = [None]*K; cnts=[0]*K; taken=0
    for i in range(0, min(len(samples), max_batches*batch_size), batch_size):
        batch=samples[i:i+batch_size]
        units=[s.unit_texts for s in batch]
        qtexts=[s.q_text for s in batch]
        slm = torch.from_numpy(np.stack([s.slm_sig for s in batch], axis=0)).to(device)
        step=torch.tensor([s.step_idx for s in batch], dtype=torch.long, device=device)
        ga = torch.tensor([s.gold_action for s in batch], dtype=torch.long, device=device)
        gm = torch.tensor([s.gold_model for s in batch], dtype=torch.long, device=device)
        model(units,qtexts,slm,step)  # forward to warm caches

        # 재인코딩: 마지막 unit 임베딩
        flat=[]; offsets=[0]
        for units_i in units:
            flat += units_i; offsets.append(offsets[-1]+len(units_i))
        Z = model.enc.encode_texts(flat, device=device)
        chunks=[]; 
        for j in range(len(units)):
            a=offsets[j]; b=offsets[j+1]
            chunks.append(Z[a:b])
        q_emb = torch.stack([chunks[j][-1] for j in range(len(units))], dim=0)

        mask = (ga!=0)
        if mask.any():
            for k in range(K):
                mk = mask & (gm==k)
                if mk.any():
                    mean_k = q_emb[mk].mean(0).detach().cpu()
                    if sums[k] is None: sums[k]=mean_k.clone()
                    else: sums[k] += mean_k
                    cnts[k] += 1
        taken += len(batch)
        if taken >= max_batches*batch_size: break
    with torch.no_grad():
        for k in range(K):
            if cnts[k]>0:
                vec = sums[k]/cnts[k]
                model.model_vecs[k].copy_(vec.to(model.model_vecs.device))

# ================= SFT =================
def lock_per_to_batch(B, pc, pd, pe, min_c=4, min_d=3, min_e=3):
    tot = pc+pd+pe
    if tot != B:
        scale = B / max(1, tot)
        pc, pd, pe = int(round(pc*scale)), int(round(pd*scale)), int(round(pe*scale))
        while pc+pd+pe < B:
            if pd <= pe: pd += 1
            else: pe += 1
        while pc+pd+pe > B:
            if pd >= pe and pd > min_d: pd -= 1
            elif pe > min_e: pe -= 1
            else: pc -= 1
    return max(min_c,pc), max(min_d,pd), max(min_e,pe)

def compute_batch_mix(B: int, freqs: Tuple[float,float,float], mode: str):
    """ legacy/data/balanced """
    pC, pD, pE = freqs
    if mode == "balanced":
        pc, pd, pe = B/3, B/3, B - 2*(B/3)
    elif mode == "data":
        pc, pd, pe = B*pC, B*pD, B*pE
    else:  # legacy
        pd = max(3, int(B*max(0.10, pD*1.25)))
        pe = max(3, int(B*max(0.10, pE*1.25)))
        pc = max(4, B - pd - pe)
        return lock_per_to_batch(B, pc, pd, pe, min_c=4, min_d=3, min_e=3)
    pc, pd, pe = int(round(pc)), int(round(pd)), int(round(pe))
    pc = max(1, pc); pd = max(1, pd); pe = max(1, pe)
    while pc+pd+pe < B:
        if pc <= pd and pc <= pe: pc += 1
        elif pd <= pe: pd += 1
        else: pe += 1
    while pc+pd+pe > B:
        if pc >= pd and pc >= pe and pc > 1: pc -= 1
        elif pd >= pe and pd > 1: pd -= 1
        elif pe > 1: pe -= 1
        else: break
    return pc, pd, pe

def build_la_bias_from_freqs(freqs: List[float], tau: float, device) -> Optional[torch.Tensor]:
    if tau is None or tau <= 0.0: return None
    p = torch.tensor(freqs, device=device).clamp_min(1e-6)
    bias = -float(tau) * p.log()
    return bias

def soft_macro_f1_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    B, C = p.size()
    losses = []
    for c in range(C):
        y = (target == c).float()
        pc = p[:, c]
        tp = (pc * y).sum()
        fp = (pc * (1.0 - y)).sum()
        fn = ((1.0 - pc) * y).sum()
        f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        losses.append(1.0 - f1)
    return torch.stack(losses).mean()

def train_sft(model, tr, dv, te, model_names, device, cfg, train_dist=None):
    no_decay = ["bias", "LayerNorm.weight"]
    bert_named = list(model.enc.bert.named_parameters())
    bert_decay    = [p for n,p in bert_named if not any(nd in n for nd in no_decay)]
    bert_no_decay = [p for n,p in bert_named if any(nd in n for nd in no_decay)]
    other_params  = [p for n,p in model.named_parameters() if not n.startswith("enc.bert")]
    opt = torch.optim.AdamW([
        {"params": bert_decay,    "lr": 2e-5, "weight_decay": 0.01},
        {"params": bert_no_decay, "lr": 2e-5, "weight_decay": 0.00},
        {"params": other_params,  "lr": 3e-4, "weight_decay": 1e-2},
    ])
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    counts={"Continue":0,"Detract":0,"Escalate":0}
    for s in tr: counts[ACTIONS[s.gold_action]] += 1
    tot = sum(counts.values())+1e-9
    freqs = [counts["Continue"]/tot, counts["Detract"]/tot, counts["Escalate"]/tot]
    inv = [1.0/(f+1e-6) for f in freqs]
    w_act = torch.tensor(inv, dtype=torch.float32, device=device); w_act = w_act / w_act.sum() * 3.0
    la_bias = build_la_bias_from_freqs(freqs, 0.5, device)

    ce_act = nn.CrossEntropyLoss(weight=w_act)
    ce_mdl = nn.CrossEntropyLoss(reduction="none")
    rdrop_lambda = 0.01
    margin_m = 0.1; margin_lambda = 0.10
    macro_gamma = 0.35
    soft_target_weight = 0.3
    gold_weight_action = 0.9
    gold_weight_model  = 0.9

    B = cfg.batch_size
    if train_dist is None:
        pC, pD, pE = freqs
    else:
        pC, pD, pE = train_dist
    dist_mode = getattr(cfg, "dist_mode", "legacy")
    per_c, per_d, per_e = compute_batch_mix(B, (pC,pD,pE), dist_mode)
    cfg.per_c, cfg.per_d, cfg.per_e = per_c, per_d, per_e
    iters_per_epoch = max(200, math.ceil(len(tr)/B))

    best_dev = {"joint": -1, "state": None, "ep": 0}

    for ep in range(1, cfg.epochs_sft+1):
        model.train(); losses=[]
        for idxs in iter_weighted_batches(tr, per_c, per_d, per_e, iters_per_epoch):
            batch=[tr[i] for i in idxs]
            units,qtexts,slm,step,ga,gm,vact,vmdl,vmdl_all=collate(batch, device)
            with autocast('cuda', enabled=use_amp):
                act1, mdl1 = model(units,qtexts,slm,step)
                act2, mdl2 = model(units,qtexts,slm,step)

            act1b = (act1 + la_bias.view(1,-1)).float()
            act2b = (act2 + la_bias.view(1,-1)).float()

            loss_act_hard = 0.5*(ce_act(act1b, ga)+ce_act(act2b, ga))
            loss_act_soft = 0.5*(soft_ce_from_masks(act1b, ga, vact, gold_weight_action) +
                                 soft_ce_from_masks(act2b, ga, vact, gold_weight_action))
            loss_act = (1.0-soft_target_weight)*loss_act_hard + soft_target_weight*loss_act_soft

            d_margin = F.relu(margin_m - (act1b[:,1]-act1b[:,0])) + F.relu(margin_m - (act2b[:,1]-act2b[:,0]))
            e_margin = F.relu(margin_m - (act1b[:,2]-act1b[:,0])) + F.relu(margin_m - (act2b[:,2]-act2b[:,0]))
            loss_margin_de = (d_margin*(ga==1).float() + e_margin*(ga==2).float())*.5

            margin_c = 0.10; margin_lambda_c = 0.05
            c_gap1 = act1b[:,0] - torch.stack([act1b[:,1], act1b[:,2]], dim=1).max(1).values
            c_gap2 = act2b[:,0] - torch.stack([act2b[:,1], act2b[:,2]], dim=1).max(1).values
            c_margin = F.relu(margin_c - c_gap1) + F.relu(margin_c - c_gap2)
            loss_margin_c = (c_margin * (ga==0).float())
            loss_margin = loss_margin_de.mean() * margin_lambda + loss_margin_c.mean() * margin_lambda_c

            mdl1m = forbid_slm_on_de(mdl1.float(), ga, slm_idx=0)
            mdl2m = forbid_slm_on_de(mdl2.float(), ga, slm_idx=0)
            mask_nc = (ga!=0).float()
            if mask_nc.sum()>0:
                rows=torch.arange(ga.size(0), device=ga.device)
                vmdl_gold_soft = vmdl_all[rows, ga, :]
                loss_m_h = ((nn.CrossEntropyLoss(reduction='none')(mdl1m, gm)+
                             nn.CrossEntropyLoss(reduction='none')(mdl2m, gm))*0.5 * mask_nc).sum() / mask_nc.sum()
                loss_m_s = 0.5*(soft_ce_from_masks(mdl1m, gm, vmdl_gold_soft, gold_weight_model) +
                                soft_ce_from_masks(mdl2m, gm, vmdl_gold_soft, gold_weight_model))
                loss_m = (1.0-soft_target_weight)*loss_m_h + soft_target_weight*loss_m_s
            else:
                loss_m = torch.tensor(0.0, device=device)

            p1 = F.log_softmax(act1b, dim=1); q1 = F.softmax(act1b, dim=1)
            p2 = F.log_softmax(act2b, dim=1); q2 = F.softmax(act2b, dim=1)
            loss_rdrop = 0.5*(F.kl_div(p1, q2, reduction="batchmean") + F.kl_div(p2, q1, reduction="batchmean")) * rdrop_lambda

            act_avg = 0.5 * (act1b + act2b)
            loss_f1 = soft_macro_f1_loss(act_avg, ga) * macro_gamma

            loss = loss_act + loss_margin + 0.7*loss_m + loss_rdrop + loss_f1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        print(f"[SFT] epoch {ep} loss={np.mean(losses):.4f}")
        a,m,j = evaluate(model, dv, model_names, device, batch_size=B, la_bias=la_bias)
        print(f"[SFT-dev@ep{ep}] action_acc={a:.4f} model_acc={m:.4f} joint={j:.4f}")

        if j > best_dev["joint"]:
            best_dev = {"joint": j, "state": copy.deepcopy(model.state_dict()), "ep": ep}

    if best_dev["state"] is not None:
        model.load_state_dict(best_dev["state"])
        print(f"[SFT] loaded best dev checkpoint @ ep{best_dev['ep']} (joint={best_dev['joint']:.4f})")

    print("== Final SFT Evaluation ==")
    a,m,j = evaluate(model, dv, model_names, device, batch_size=B, la_bias=la_bias)
    print(f"[SFT-dev] action_acc={a:.4f} model_acc={m:.4f} joint={j:.4f}")
    a,m,j = evaluate(model, te, model_names, device, batch_size=B, la_bias=la_bias)
    print(f"[SFT-test] action_acc={a:.4f} model_acc={m:.4f} joint={j:.4f}")

    return la_bias  # SFT 시 사용한 기본 LA를 RL/GRPO에서도 사용

# ================= Bias calibration (Gold metric) =================
@torch.no_grad()
def _gold_metric_with_bias(model, samples, device, la_bias: Optional[torch.Tensor], target: str = "joint", batch_size: int = 32):
    metrics = compute_metrics(model, samples, [f"m{i}" for i in range(model.mdl_head.out_features)], device, batch_size=batch_size, la_bias=la_bias)
    return metrics["gold_joint_acc"] if target == "joint" else metrics["gold_action_acc"]

@torch.no_grad()
def calibrate_for_gold(model, samples, device,
                       base_bias: Optional[torch.Tensor] = None,
                       search_lo: float = -1.0, search_hi: float = 1.0, search_step: float = 0.1,
                       target: str = "joint") -> torch.Tensor:
    model.eval()
    grid = np.arange(search_lo, search_hi + 1e-9, search_step)
    best = {"score": -1.0, "bias": None, "bd": 0.0, "be": 0.0}
    base = base_bias.clone().to(device) if isinstance(base_bias, torch.Tensor) else None
    zero = torch.zeros(3, device=device)
    for bd in grid:
        for be in grid:
            add = zero.clone(); add[1]=float(bd); add[2]=float(be)
            cur = (base + add) if base is not None else add
            score = _gold_metric_with_bias(model, samples, device, cur, target=target, batch_size=32)
            if score > best["score"]:
                best = {"score": score, "bias": cur.detach().clone(), "bd": bd, "be": be}
    print(f"[GOLD-CAL] target={target} Δbias(D,E)=({best['bd']:+.2f},{best['be']:+.2f}) -> dev {target}={best['score']:.4f}")
    return best["bias"]

# ================= GRPO =================
def _sample_from_logits(logits: torch.Tensor, temp: float = 1.0):
    p = F.softmax(logits / max(1e-6, temp), dim=1).clamp(min=1e-8)
    dist = torch.distributions.Categorical(p)
    a = dist.sample()
    logp = dist.log_prob(a)
    return a, logp, p

def _reward_components(pa, pm, ga, gm, vact, vmdl_all):
    """
    보상:
      +1.0 if action==gold
      else +0.5 if action acceptable
      +0.7 if (a!=C and model==gold_model)
      else +0.35 if (a!=C and route_ok)
      -0.6 if (gold in {D,E} and predicted is C)
      -0.3 if (a!=C and route invalid)
      +0.2 if (ga==C and pa==C)
    """
    B = pa.size(0)
    rows = torch.arange(B, device=pa.device)
    is_nc = (pa != 0)
    same_act = (pa == ga).float()
    accept = vact[rows, pa].float()
    base = same_act + (1.0 - same_act) * 0.5 * accept
    route_ok = vmdl_all[rows, pa, pm].float() * is_nc.float()
    same_model = ((pm == gm).float()) * is_nc.float()
    add_route = 0.7 * same_model + 0.35 * (1.0 - same_model) * route_ok
    false_continue = ((ga != 0) & (pa == 0)).float()
    invalid_route = ((is_nc) & (vmdl_all[rows, pa, pm] < 0.5)).float()
    reward = base + add_route - 0.6 * false_continue - 0.3 * invalid_route
    reward = reward + 0.2 * ((ga == 0) & (pa == 0)).float()
    return reward

@torch.no_grad()
def precompute_anchor_logits(anchor_model, samples: List[TrainSample], device, batch_size=32):
    anchor_model.eval()
    N = len(samples)
    act_buf = torch.zeros((N, 3), dtype=torch.float16)
    mdl_buf = torch.zeros((N, anchor_model.mdl_head.out_features), dtype=torch.float16)
    for idxs in iter_eval_batches(N, batch_size):
        batch=[samples[i] for i in idxs]
        units,qtexts,slm,step,_,_,_,_,_=collate(batch, device)
        act_l, mdl_l = anchor_model(units,qtexts,slm,step)
        act_buf[idxs] = act_l.detach().cpu().to(torch.float16)
        mdl_buf[idxs] = mdl_l.detach().cpu().to(torch.float16)
    return act_buf, mdl_buf

def _repeat_batch_for_group(units, qtexts, slm, step, ga, gm, vact, vmdl_all, G: int):
    def _rep_list(L):
        out=[]
        for x in L: out.extend([x]*G)
        return out
    units_rep  = _rep_list(units)
    qtexts_rep = _rep_list(qtexts)
    slm_rep    = slm.repeat_interleave(G, dim=0)
    step_rep   = step.repeat_interleave(G, dim=0)
    ga_rep     = ga.repeat_interleave(G, dim=0)
    gm_rep     = gm.repeat_interleave(G, dim=0)
    vact_rep   = vact.repeat_interleave(G, dim=0)
    vmdl_rep   = vmdl_all.repeat_interleave(G, dim=0)
    return units_rep, qtexts_rep, slm_rep, step_rep, ga_rep, gm_rep, vact_rep, vmdl_rep

def _group_advantages(rewards: torch.Tensor, G: int, baseline: str="loo",
                      topk_frac: float=1.0, norm: str="group"):
    """
    rewards: [B*G] → [B,G]로 reshape
    baseline: 'loo' | 'mean' | 'topk'
    norm: 'group' | 'batch' | 'none'
    """
    N = rewards.numel()
    if G <= 0: raise ValueError(f"G must be >=1, got {G}")
    if N % G != 0:
        B_eff = N // G
        rewards = rewards[:B_eff * G]
    B = rewards.numel() // G
    R = rewards.reshape(B, G)
    if baseline == "loo" and G > 1:
        sumR = R.sum(dim=1, keepdim=True)
        base = (sumR - R) / max(1, G - 1)
    elif baseline == "mean":
        base = R.mean(dim=1, keepdim=True).expand_as(R)
    else:
        k = max(1, int(round(G * float(topk_frac))))
        topk_vals, _ = torch.topk(R, k, dim=1, largest=True, sorted=False)
        base = topk_vals.mean(dim=1, keepdim=True).expand_as(R)
    A = (R - base)
    if norm == "group":
        std = R.std(dim=1, keepdim=True).clamp_min(1e-6)
        A = (A / std)
    elif norm == "batch":
        stdb = rewards.std().clamp_min(1e-6)
        A = (A / stdb)
    return A.reshape(B * G).detach()

def train_grpo(model, tr, dv, te, model_names, device, cfg, la_bias_eval: Optional[torch.Tensor],
               anchor_precomputed=None, args=None):
    """ GRPO (Group Relative Policy Optimization) """
    assert anchor_precomputed is not None, "anchor_precomputed가 필요합니다 (SFT reference 고정 로짓)."
    anchor_act_all, anchor_mdl_all = anchor_precomputed  # CPU float16

    no_decay = ["bias", "LayerNorm.weight"]
    bert_named = list(model.enc.bert.named_parameters())
    bert_decay    = [p for n,p in bert_named if not any(nd in n for nd in no_decay)]
    bert_no_decay = [p for n,p in bert_named if any(nd in n for nd in no_decay)]
    other_params  = [p for n,p in model.named_parameters() if not n.startswith("enc.bert")]
    opt = torch.optim.AdamW([
        {"params": bert_decay,    "lr": 2e-5, "weight_decay": 0.01},
        {"params": bert_no_decay, "lr": 2e-5, "weight_decay": 0.00},
        {"params": other_params,  "lr": 3e-4, "weight_decay": 1e-2},
    ])
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # 클래스 불균형 보상 가중 (평균 ≈ 1)
    counts = {"Continue":0,"Detract":0,"Escalate":0}
    for s in tr: counts[ACTIONS[s.gold_action]] += 1
    tot = sum(counts.values()) + 1e-9
    freqs = np.array([counts["Continue"]/tot, counts["Detract"]/tot, counts["Escalate"]/tot], dtype=np.float32)
    inv = 1.0 / np.clip(freqs, 1e-6, None)
    cls_w = torch.tensor(inv / inv.mean(), device=device).float()

    # 약한 CE 안정화
    ce_act_guided = nn.CrossEntropyLoss(weight=torch.tensor(inv, device=device).float())
    ce_mdl_guided = nn.CrossEntropyLoss(reduction="none")

    # 하이퍼
    G = int(getattr(args, "group_size", 4) if args is not None else 4)
    T_act, T_mdl = 0.8, 0.8
    LAMBDA_KL  = float(getattr(args, "kl_coef", 0.25) if args is not None else 0.25)
    LAMBDA_CE  = float(getattr(args, "ce_coef", 0.10) if args is not None else 0.10)
    LAMBDA_ENT = float(getattr(args, "ent_coef", 0.01) if args is not None else 0.01)
    BASELINE   = str(getattr(args, "baseline", "loo") if args is not None else "loo")
    TOPK_FRAC  = float(getattr(args, "topk_frac", 1.0) if args is not None else 1.0)
    R_NORM     = str(getattr(args, "reward_norm", "group") if args is not None else "group")

    iters_per_epoch = max(120, math.ceil(len(tr)/cfg.batch_size))
    per_c, per_d, per_e = cfg.per_c, cfg.per_d, cfg.per_e

    best_dev = {"joint": -1, "state": None, "ep": 0}

    for ep in range(1, cfg.epochs_rl+1):
        model.train()
        returns=[]; kls=[]
        for idxs in iter_weighted_batches(tr, per_c, per_d, per_e, iters_per_epoch):
            batch=[tr[i] for i in idxs]
            units,qtexts,slm,step,ga,gm,vact,_,vmdl_all = collate(batch, device)
            B = ga.size(0)

            # 그룹 반복
            units_r, qtexts_r, slm_r, step_r, ga_r, gm_r, vact_r, vmdl_r = _repeat_batch_for_group(
                units, qtexts, slm, step, ga, gm, vact, vmdl_all, G
            )

            with autocast('cuda', enabled=use_amp):
                act_logits, mdl_logits = model(units_r, qtexts_r, slm_r, step_r)
            if la_bias_eval is not None:
                act_logits = act_logits + la_bias_eval.view(1, -1)

            # 샘플링
            a, logp_a, _ = _sample_from_logits(act_logits, T_act)
            mdl_logits_masked = forbid_slm_on_de(mdl_logits, a, slm_idx=0)
            m, logp_m, _ = _sample_from_logits(mdl_logits_masked, T_mdl)
            logp_total = logp_a + (a != 0).float() * logp_m  # [B*G]

            # 보상
            r = _reward_components(a, m, ga_r, gm_r, vact_r, vmdl_r)  # [B*G]
            r = r * cls_w[ga_r]  # 클래스 가중

            # 그룹 상대 어드밴티지
            A = _group_advantages(r, G, baseline=BASELINE, topk_frac=TOPK_FRAC, norm=R_NORM)

            # Policy gradient
            pg_loss = -(A * logp_total).mean()

            # KL(π || π_ref=SFT) — anchor precomputed 로짓 사용
            a0_logits = anchor_act_all[idxs].to(device=device, dtype=torch.float32).repeat_interleave(G, dim=0)
            m0_logits = anchor_mdl_all[idxs].to(device=device, dtype=torch.float32).repeat_interleave(G, dim=0)
            if la_bias_eval is not None:
                a0_logits = a0_logits + la_bias_eval.view(1, -1)

            kl_a = kl_penalty(act_logits.float(), a0_logits.float())

            # 모델 KL은 금라벨 기준 SLM 금지
            mdl_logits_gold_mask = forbid_slm_on_de(mdl_logits.float(), ga_r, slm_idx=0)
            m0_logits_gold_mask  = forbid_slm_on_de(m0_logits.float(),  ga_r, slm_idx=0)
            kl_m = kl_penalty(mdl_logits_gold_mask, m0_logits_gold_mask)
            kl_loss = kl_a + kl_m

            # 약한 CE (안정화)
            ce_a = ce_act_guided(act_logits.float(), ga_r)
            mask_nc = (ga_r != 0).float()
            ce_m = ce_mdl_guided(mdl_logits_gold_mask, gm_r)
            ce_m = (ce_m * mask_nc).sum() / (mask_nc.sum() + 1e-6)
            ce_loss = ce_a + ce_m

            # 엔트로피 보너스(최대화 → 손실에서 빼기)
            ent_a = entropy_bonus(act_logits.float())
            if (a != 0).any():
                ent_m = entropy_bonus(mdl_logits_masked[(a != 0)].float())
            else:
                ent_m = torch.tensor(0.0, device=device)
            ent = ent_a + ent_m

            loss = pg_loss + LAMBDA_KL * kl_loss + LAMBDA_CE * ce_loss - LAMBDA_ENT * ent

            opt.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            returns.append(float(r.mean().item()))
            kls.append(float(kl_loss.item()))

        # Dev 평가
        a_acc, m_acc, j_acc = evaluate(model, dv, model_names, device, batch_size=cfg.batch_size, la_bias=la_bias_eval)
        print(f"[GRPO] epoch {ep} avg_return={np.mean(returns):.4f} avg_KL={np.mean(kls):.4f}")
        print(f"[GRPO-dev@ep{ep}] action_acc={a_acc:.4f} model_acc={m_acc:.4f} joint={j_acc:.4f}")

        if j_acc > best_dev["joint"]:
            best_dev = {"joint": j_acc, "state": copy.deepcopy(model.state_dict()), "ep": ep}

    if best_dev["state"] is not None:
        model.load_state_dict(best_dev["state"])
        print(f"[GRPO] loaded best dev checkpoint @ epoch {best_dev['ep']} (joint={best_dev['joint']:.4f})")

    print("== Final GRPO Evaluation ==")
    a_acc, m_acc, j_acc = evaluate(model, dv, model_names, device, batch_size=cfg.batch_size, la_bias=la_bias_eval)
    print(f"[GRPO-dev] action_acc={a_acc:.4f} model_acc={m_acc:.4f} joint={j_acc:.4f}")
    a_acc, m_acc, j_acc = evaluate(model, te, model_names, device, batch_size=cfg.batch_size, la_bias=la_bias_eval)
    print(f"[GRPO-test] action_acc={a_acc:.4f} model_acc={m_acc:.4f} joint={j_acc:.4f}")

# ================= Config & CLI =================
@dataclass
class SimpleCfg:
    batch_size: int = 16
    epochs_sft: int = 8
    epochs_rl: int  = 6
    dist_mode: str = "legacy"
    # filled later
    per_c: int = 0
    per_d: int = 0
    per_e: int = 0

def parse_args():
    ap = argparse.ArgumentParser(description="GRU-seq Router (SFT + GRPO)")
    ap.add_argument("--data", nargs="+", required=True, help="학습 jsonl 파일(들)")
    ap.add_argument("--models", nargs="+", default=["SLM","Qwen/Qwen2.5-7B-Instruct","meta-llama/Llama-3.1-8B-Instruct","Qwen/Qwen2.5-14B-Instruct"])
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--freeze_bert", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs_sft", type=int, default=8)
    ap.add_argument("--epochs_rl", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", default="ckpts", help="ckpt 저장 폴더")
    ap.add_argument("--save_name", default="router_rl.pt", help="최종 RL 베스트 ckpt 파일명")
    ap.add_argument("--sft_name", default=None, help="SFT ckpt 파일명(미지정시 자동)")
    ap.add_argument("--label_source", choices=["auto","sample_gold","teacher"], default="auto")
    ap.add_argument("--dist_mode", choices=["legacy","data","balanced"], default="legacy")

    # GRPO options (+ 호환성 위해 --rl_alg 유지)
    ap.add_argument("--rl_alg", choices=["grpo"], default="grpo")
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--baseline", choices=["loo","mean","topk"], default="loo")
    ap.add_argument("--topk_frac", type=float, default=1.0)
    ap.add_argument("--reward_norm", choices=["none","group","batch"], default="group")
    ap.add_argument("--kl_coef", type=float, default=0.25)
    ap.add_argument("--ce_coef", type=float, default=0.10)
    ap.add_argument("--ent_coef", type=float, default=0.01)

    return ap.parse_args()

# ================= Main =================
def main():
    args = parse_args()
    if getattr(args, "rl_alg", "grpo") != "grpo":
        print("[warn] --rl_alg은 grpo만 지원합니다. grpo로 강제 설정합니다.")
    set_seed(args.seed)
    device = torch.device(args.device)

    raw = read_jsonl(args.data)
    allowed_models = args.models
    print(f"[Vocab] actions= {ACTIONS}")
    print(f"[Vocab] models= {allowed_models}  (SLM_idx=0)")

    eps = parse_episodes(raw, allowed_models)
    tr, dv, te = build_train_samples(
        eps, allowed_models,
        split_ratios=(0.7,0.15,0.15),
        seed=args.seed,
        label_source=args.label_source
    )

    # feature stats
    use_aux, use_slm = True, True
    aux_mean, aux_std, slm_mean, slm_std = compute_feature_stats(tr, use_aux, use_slm)

    # model
    model = GRURouter(
        num_models=len(allowed_models),
        aux_dim=aux_dim_reduced() if use_aux else 0,
        slm_sig_dim=len(tr[0].slm_sig) if (use_slm and len(tr)>0) else 0,
        bert_name=args.bert, max_len=args.max_len,
        freeze_bert=bool(args.freeze_bert), bert_local_files_only=False
    ).to(device)
    model.set_feature_stats(aux_mean.to(device), aux_std.to(device), slm_mean.to(device), slm_std.to(device))

    # init heads by label stats + lightweight vec init
    init_biases_from_label_stats(model, tr, allowed_models, device)
    initialize_from_data(model, tr, device)

    # SFT
    cfg = SimpleCfg(batch_size=args.batch_size, epochs_sft=args.epochs_sft, epochs_rl=args.epochs_rl, dist_mode=args.dist_mode)
    la_bias_sft = train_sft(model, tr, dv, te, allowed_models, device, cfg)
    sft_state = copy.deepcopy(model.state_dict())

    # dev 기준 gold_joint 최적화로 Δbias 튜닝
    la_bias_tuned = calibrate_for_gold(model, dv, device, base_bias=la_bias_sft, target="joint")

    # RL용 Anchor 사전계산 (SFT 고정본)
    anchor_model = copy.deepcopy(model).eval()
    for p in anchor_model.parameters(): p.requires_grad_(False)
    anchor_pre = precompute_anchor_logits(anchor_model, tr, device, batch_size=cfg.batch_size)
    del anchor_model

    # GRPO
    train_grpo(model, tr, dv, te, allowed_models, device, cfg,
               la_bias_eval=la_bias_tuned, anchor_precomputed=anchor_pre, args=args)

    # --- 저장 ---
    os.makedirs(args.save_dir, exist_ok=True)
    # SFT
    sft_name = args.sft_name if args.sft_name else "router_sft.pt"
    torch.save({
        "state_dict": sft_state,
        "feature_stats": {
            "aux_mean": aux_mean.tolist(),
            "aux_std":  aux_std.tolist(),
            "slm_mean": slm_mean.tolist(),
            "slm_std":  slm_std.tolist(),
        },
        "la_bias": (la_bias_sft.tolist() if la_bias_sft is not None else None),
        "model_names": allowed_models,
        "bert": args.bert,
        "max_len": args.max_len,
        "freeze_bert": int(args.freeze_bert),
        "tag": "SFT-best"
    }, os.path.join(args.save_dir, sft_name))
    print(f"[SAVE] SFT-best -> {os.path.join(args.save_dir, sft_name)}")

    # RL
    final_bundle = {
        "state_dict": model.state_dict(),
        "feature_stats": {
            "aux_mean": aux_mean.tolist(),
            "aux_std":  aux_std.tolist(),
            "slm_mean": slm_mean.tolist(),
            "slm_std":  slm_std.tolist(),
        },
        "la_bias": (la_bias_tuned.tolist() if la_bias_tuned is not None else None),
        "model_names": allowed_models,
        "bert": args.bert,
        "max_len": args.max_len,
        "freeze_bert": int(args.freeze_bert),
        "tag": "RL-best"
    }
    save_path = os.path.join(args.save_dir, args.save_name)
    torch.save(final_bundle, save_path)
    print(f"[SAVE] RL-best -> {save_path}")

if __name__ == "__main__":
    main()
