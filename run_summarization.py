"""
Attention diffusion for abstractive summarization.

Pipeline:
  article → N_SENT sentence embeddings  X [B, N_SENT, D]
  ↓  FreeDecoder (seq2seq, teacher-forced)
  A_0 [B, L_SUM, N_SENT]   (cross-attention from last decoder layer)
  ↓  GNNDenoiser
  learns P(A | X)
  ↓  GuidedDecoder
  decodes summary given A_0 from denoiser (vs. oracle A_0 vs. free)

Key ablations at test time:
  free      : FreeDecoder, own cross-attention (baseline)
  guided-oracle : GuidedDecoder + true A_0
  guided-K1 : GuidedDecoder + 1-step denoised A_0
  guided-K8 : GuidedDecoder + 8-step denoised A_0
"""

import os, sys, json, re, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Hypers ─────────────────────────────────────────────────────────────────
D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 3
N_SENT     = 8      # sentences per article
SENT_LEN   = 24     # words per sentence (truncated/padded)
L_SUM      = 40     # max summary tokens
VOCAB_SIZE = 8000
BATCH      = 32
LR         = 3e-4
EPOCHS_FREE     = 50
EPOCHS_DEN      = 40
EPOCHS_GUIDED   = 40
TRAIN_N    = 5000
VAL_N      = 500
TEST_N     = 500
N_MP       = 2
K_LIST     = [1, 2, 4, 8]
SEEDS      = [42, 7, 13]
RESULTS_DIR = "/home/faldabbas/myproject/attn_diffusion_gen/results"

# ── Tokenizer ──────────────────────────────────────────────────────────────
class Tokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def build(self, texts, max_vocab=VOCAB_SIZE):
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        vocab += [w for w, _ in counts.most_common(max_vocab - 4)]
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.vocab = vocab

    def enc(self, text, max_len, bos=False, eos=True):
        toks = [self.w2i.get(w.lower(), self.UNK) for w in text.split()]
        if bos: toks = [self.BOS] + toks
        if eos: toks = toks + [self.EOS]
        toks = toks[:max_len]
        toks += [self.PAD] * (max_len - len(toks))
        return toks

    def dec(self, ids):
        words = []
        for i in ids:
            if i in {self.PAD, self.BOS}: continue
            if i == self.EOS: break
            words.append(self.vocab[i])
        return " ".join(words)

# ── Data ───────────────────────────────────────────────────────────────────
def split_sentences(text, n):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if len(s.split()) >= 3][:n]
    while len(sents) < n:
        sents.append(sents[-1] if sents else "empty")
    return sents

class SumDataset(Dataset):
    def __init__(self, records, tok):
        self.items = []
        for r in records:
            article = r["document"] if "document" in r else r["article"]
            summary = r["summary"] if "summary" in r else r["highlights"].split("\n")[0]
            sents = split_sentences(article, N_SENT)
            sent_ids = [tok.enc(s, SENT_LEN, eos=False) for s in sents]
            sum_ids  = tok.enc(summary, L_SUM, bos=True, eos=True)
            self.items.append({
                "sent": torch.tensor(sent_ids, dtype=torch.long),  # [N_SENT, SENT_LEN]
                "sum":  torch.tensor(sum_ids,  dtype=torch.long),  # [L_SUM]
                "ref":  summary,
            })

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    return {
        "sent": torch.stack([b["sent"] for b in batch]),
        "sum":  torch.stack([b["sum"]  for b in batch]),
        "ref":  [b["ref"] for b in batch],
    }

def load_data():
    from datasets import load_dataset
    print("Loading XSum...")
    ds = load_dataset("xsum")
    train = list(ds["train"].select(range(TRAIN_N)))
    val   = list(ds["validation"].select(range(VAL_N)))
    test  = list(ds["test"].select(range(TEST_N)))
    return train, val, test

# ── Models ─────────────────────────────────────────────────────────────────
class SentEncoder(nn.Module):
    """Sentence → D via avg of word embeddings."""
    def __init__(self, V, D):
        super().__init__()
        self.embed = nn.Embedding(V, D, padding_idx=0)
        self.proj  = nn.Linear(D, D)

    def forward(self, x):
        # x: [B, N_SENT, SENT_LEN]
        mask = (x != 0).float().unsqueeze(-1)
        e    = self.embed(x)                                   # [B, N, L, D]
        avg  = (e * mask).sum(2) / mask.sum(2).clamp(min=1)   # [B, N, D]
        return F.gelu(self.proj(avg))

class DecoderLayer(nn.Module):
    def __init__(self, D, H, dropout=0.1):
        super().__init__()
        self.sa   = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.ca   = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D))
        self.ln1  = nn.LayerNorm(D)
        self.ln2  = nn.LayerNorm(D)
        self.ln3  = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask=None, tgt_pad=None, mem_bias=None, ret_attn=False):
        # Self-attention
        xn = self.ln1(x)
        sa_out, _ = self.sa(xn, xn, xn, attn_mask=tgt_mask, key_padding_mask=tgt_pad)
        x = x + self.drop(sa_out)
        # Cross-attention
        xn = self.ln2(x)
        if mem_bias is not None:
            # mem_bias: [B, L, N_SENT] — add as attention logit offset
            ca_out, attn_w = self.ca(xn, mem, mem, attn_mask=None, need_weights=True,
                                     average_attn_weights=True)
            # We'll handle the bias post-hoc below; this is the free path
            _ = attn_w  # unused when bias provided — see guided path
        if ret_attn:
            ca_out, attn_w = self.ca(xn, mem, mem, attn_mask=None,
                                     need_weights=True, average_attn_weights=True)
        else:
            ca_out, attn_w = self.ca(xn, mem, mem, need_weights=False)
        x = x + self.drop(ca_out)
        x = x + self.drop(self.ff(self.ln3(x)))
        return (x, attn_w) if ret_attn else x

class GuidedDecoderLayer(nn.Module):
    """Cross-attention replaced by A_0-biased version."""
    def __init__(self, D, H, dropout=0.1):
        super().__init__()
        self.sa   = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.q    = nn.Linear(D, D)
        self.k    = nn.Linear(D, D)
        self.v    = nn.Linear(D, D)
        self.o    = nn.Linear(D, D)
        self.ff   = nn.Sequential(nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D))
        self.ln1  = nn.LayerNorm(D)
        self.ln2  = nn.LayerNorm(D)
        self.ln3  = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)
        self.scale = D ** -0.5
        self.H = H
        self.D = D

    def forward(self, x, mem, attn_plan, tgt_mask=None, tgt_pad=None):
        # x: [B,L,D]  mem: [B,N,D]  attn_plan: [B,L,N]  (soft attention guide)
        B, L, D = x.shape
        xn = self.ln1(x)
        sa_out, _ = self.sa(xn, xn, xn, attn_mask=tgt_mask, key_padding_mask=tgt_pad)
        x = x + self.drop(sa_out)

        xn = self.ln2(x)
        Q = self.q(xn)                                     # [B,L,D]
        K = self.k(mem)                                    # [B,N,D]
        V = self.v(mem)                                    # [B,N,D]
        logits = torch.bmm(Q, K.transpose(-1,-2)) * self.scale  # [B,L,N]
        # blend: learned logits + plan (plan is in prob space → convert to logits via log)
        plan_logit = torch.log(attn_plan.clamp(min=1e-6))
        logits = logits + plan_logit
        weights = F.softmax(logits, dim=-1)
        ctx = torch.bmm(weights, V)                        # [B,L,D]
        x = x + self.drop(self.o(ctx))
        x = x + self.drop(self.ff(self.ln3(x)))
        return x

class FreeDecoder(nn.Module):
    def __init__(self, V, D, H, NL, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos   = nn.Embedding(L_SUM+2, D)
        self.layers = nn.ModuleList([DecoderLayer(D, H, dropout) for _ in range(NL)])
        self.head  = nn.Linear(D, V)
        self.NL = NL

    def forward(self, src, tgt, ret_last_attn=False):
        B, L = tgt.shape
        x = self.embed(tgt) + self.pos(torch.arange(L, device=tgt.device))
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == 0)
        attn_w = None
        for i, layer in enumerate(self.layers):
            last = (i == self.NL - 1) and ret_last_attn
            out = layer(x, src, tgt_mask=causal, tgt_pad=tgt_pad, ret_attn=last)
            if last:
                x, attn_w = out
            else:
                x = out
        logits = self.head(x)
        if ret_last_attn:
            return logits, attn_w
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=L_SUM, bos=1, eos=2):
        B = src.shape[0]
        toks = torch.full((B, 1), bos, device=src.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.forward(src, toks)
            next_t = logits[:, -1].argmax(-1)
            toks = torch.cat([toks, next_t.unsqueeze(1)], dim=1)
            done |= (next_t == eos)
            if done.all(): break
        return toks

class GuidedDecoder(nn.Module):
    def __init__(self, V, D, H, NL, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(V, D)
        self.pos    = nn.Embedding(L_SUM+2, D)
        self.layers = nn.ModuleList([GuidedDecoderLayer(D, H, dropout) for _ in range(NL)])
        self.head   = nn.Linear(V, V) if False else nn.Linear(D, V)

    def forward(self, src, tgt, attn_plan):
        B, L = tgt.shape
        x = self.embed(tgt) + self.pos(torch.arange(L, device=tgt.device))
        causal  = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == 0)
        for layer in self.layers:
            x = layer(x, src, attn_plan[:, :L, :], tgt_mask=causal, tgt_pad=tgt_pad)
        return self.head(x)

    @torch.no_grad()
    def generate(self, src, attn_plan, max_len=L_SUM, bos=1, eos=2):
        B = src.shape[0]
        toks = torch.full((B, 1), bos, device=src.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for step in range(max_len - 1):
            logits = self.forward(src, toks, attn_plan)
            next_t = logits[:, -1].argmax(-1)
            toks = torch.cat([toks, next_t.unsqueeze(1)], dim=1)
            done |= (next_t == eos)
            if done.all(): break
        return toks

# ── GNN Denoiser (rectangular) ─────────────────────────────────────────────
class GNNDenoiser(nn.Module):
    def __init__(self, D, n_src=N_SENT, n_tgt=L_SUM, n_mp=N_MP):
        super().__init__()
        self.row_proj   = nn.Linear(n_src, D)
        self.alpha_emb  = nn.Sequential(nn.Linear(1, D), nn.GELU(), nn.Linear(D, D))
        self.msg_projs  = nn.ModuleList([nn.Linear(D, D) for _ in range(n_mp)])
        self.mp_norms   = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_mp)])
        self.q_proj     = nn.Linear(D, D)
        self.k_proj     = nn.Linear(D, D)
        self.scale      = D ** -0.5

    def forward(self, X_src, A_t, alpha):
        # X_src:[B,N,D]  A_t:[B,L,N]  alpha:[B]
        B = A_t.shape[0]
        a_e = self.alpha_emb(alpha.view(B, 1, 1))        # [B,1,D]
        h   = F.gelu(self.row_proj(A_t)) + a_e           # [B,L,D]
        for mp, ln in zip(self.msg_projs, self.mp_norms):
            msgs = torch.bmm(A_t, mp(X_src))             # [B,L,D]
            h    = ln(h + msgs)
        Q = F.normalize(self.q_proj(h), dim=-1)           # [B,L,D]
        K = self.k_proj(X_src)                            # [B,N,D]
        return torch.bmm(Q, K.transpose(-1,-2)) / self.scale  # [B,L,N]

# ── Noise ──────────────────────────────────────────────────────────────────
def noise_attn(A0, alpha):
    """Interpolate A_0 toward uniform. A0:[B,L,N] alpha:[B]."""
    B, L, N = A0.shape
    uniform = torch.full_like(A0, 1.0 / N)
    a = alpha.view(B, 1, 1)
    return a * uniform + (1 - a) * A0

@torch.no_grad()
def multi_step_denoise(denoiser, X_src, n_steps, B=None, device=None):
    if B is None: B = X_src.shape[0]
    if device is None: device = X_src.device
    N = X_src.shape[1]
    # start from uniform
    A = torch.full((B, L_SUM, N), 1.0 / N, device=device)
    alphas = torch.linspace(1.0, 0.0, n_steps + 1, device=device)[1:]  # from 1→0
    for alpha in alphas:
        logits = denoiser(X_src, A, alpha.expand(B))
        A = F.softmax(logits, dim=-1)
    return A

# ── ROUGE helper ───────────────────────────────────────────────────────────
def rouge1_f1(hyp, ref):
    h = set(hyp.lower().split())
    r = set(ref.lower().split())
    if not h or not r: return 0.0
    ov = len(h & r)
    p = ov / len(h)
    rec = ov / len(r)
    if p + rec == 0: return 0.0
    return 2 * p * rec / (p + rec)

def eval_rouge(preds, refs):
    return float(np.mean([rouge1_f1(p, r) for p, r in zip(preds, refs)]))

def eval_rouge_best_of_n(preds, refs_list):
    """Best-of-N-reference ROUGE-1: max over each example's reference set,
    then mean over examples. refs_list[i] is the list of references for
    example i (e.g. 5 MS-COCO captions)."""
    return float(np.mean([
        max(rouge1_f1(p, r) for r in refs) for p, refs in zip(preds, refs_list)
    ]))

# ── Training helpers ───────────────────────────────────────────────────────
def train_free(enc, dec, loader, n_epochs, device):
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        enc.train(); dec.train()
        tot = 0; n = 0
        for batch in loader:
            sent = batch["sent"].to(device)   # [B,N,SENT_LEN]
            s    = batch["sum"].to(device)    # [B,L]
            X = enc(sent)                     # [B,N,D]
            logits = dec(X, s[:, :-1])        # [B,L-1,V]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   s[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  free_loss={tot/n:.4f}")

@torch.no_grad()
def extract_A0(enc, dec, loader, device):
    enc.eval(); dec.eval()
    A0s, Xs, refs = [], [], []
    for batch in loader:
        sent = batch["sent"].to(device)
        s    = batch["sum"].to(device)
        X = enc(sent)
        _, attn_w = dec(X, s[:, :-1], ret_last_attn=True)  # [B,L-1,N]
        # pad to L_SUM
        B, L1, N = attn_w.shape
        pad = torch.zeros(B, L_SUM - L1, N, device=device)
        attn_full = torch.cat([attn_w, pad], dim=1)  # [B,L_SUM,N]
        A0s.append(attn_full.cpu())
        Xs.append(X.cpu())
        refs += batch["ref"]
    return torch.cat(A0s), torch.cat(Xs), refs

class A0Dataset(Dataset):
    def __init__(self, A0, X, sum_ids):
        self.A0 = A0; self.X = X; self.sum_ids = sum_ids

    def __len__(self): return len(self.A0)
    def __getitem__(self, i):
        return self.A0[i], self.X[i], self.sum_ids[i]

def train_denoiser(denoiser, A0_dataset, n_epochs, device):
    loader = DataLoader(A0_dataset, batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(denoiser.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0; n = 0
        for A0, X, _ in loader:
            A0 = A0.to(device); X = X.to(device)
            B = A0.shape[0]
            alpha = torch.rand(B, device=device)
            A_t = noise_attn(A0, alpha)
            logits = denoiser(X, A_t, alpha)  # [B,L,N]
            loss = F.cross_entropy(
                logits.view(-1, N_SENT),
                A0.argmax(-1).view(-1)
            )
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  den_loss={tot/n:.4f}")

def train_guided(enc, guided_dec, A0_dataset, tok, n_epochs, device):
    params = list(guided_dec.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    # rebuild sum_ids from A0Dataset (sum_ids are tensors)
    for ep in range(1, n_epochs + 1):
        guided_dec.train()
        loader = DataLoader(A0_dataset, batch_size=BATCH, shuffle=True)
        tot = 0; cnt = 0
        for A0, X, sum_ids in loader:
            A0 = A0.to(device); X = X.to(device); sum_ids = sum_ids.to(device)
            # A_plan = softmax over A0 → probability plan
            plan = F.softmax(A0 * 10, dim=-1)  # sharpen for training
            logits = guided_dec(X, sum_ids[:, :-1], plan)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   sum_ids[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); cnt += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  guided_loss={tot/cnt:.4f}")

# ── Evaluation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(enc, free_dec, guided_dec, denoiser, test_loader, tok, device):
    enc.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()

    free_preds, oracle_preds, refs = [], [], []
    k_preds = {k: [] for k in K_LIST}

    for batch in test_loader:
        sent = batch["sent"].to(device)
        s    = batch["sum"].to(device)
        X    = enc(sent)
        B    = X.shape[0]

        # Free generation
        toks = free_dec.generate(X)
        for row in toks.cpu().tolist():
            free_preds.append(tok.dec(row))

        # Oracle A_0
        _, attn_w = free_dec(X, s[:, :-1], ret_last_attn=True)  # [B,L-1,N]
        L1 = attn_w.shape[1]
        pad = torch.zeros(B, L_SUM - L1, N_SENT, device=device)
        A_oracle = torch.cat([attn_w, pad], dim=1)
        plan_oracle = F.softmax(A_oracle * 10, dim=-1)
        toks_or = guided_dec.generate(X, plan_oracle)
        for row in toks_or.cpu().tolist():
            oracle_preds.append(tok.dec(row))

        refs += batch["ref"]

        # Denoiser K steps
        for k in K_LIST:
            A_den = multi_step_denoise(denoiser, X, k)
            plan_den = F.softmax(A_den * 10, dim=-1)
            toks_k = guided_dec.generate(X, plan_den)
            for row in toks_k.cpu().tolist():
                k_preds[k].append(tok.dec(row))

    return {
        "free":    eval_rouge(free_preds,   refs),
        "oracle":  eval_rouge(oracle_preds, refs),
        **{f"K{k}": eval_rouge(k_preds[k], refs) for k in K_LIST},
    }

# ── Recovery test ──────────────────────────────────────────────────────────
@torch.no_grad()
def recovery_test(denoiser, A0_tensor, X_tensor, n_samples=256, device=DEVICE):
    denoiser.eval()
    idx = torch.randperm(len(A0_tensor))[:n_samples]
    A0 = A0_tensor[idx].to(device)
    X  = X_tensor[idx].to(device)
    results = {}
    for alpha_v in [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]:
        alpha = torch.full((n_samples,), alpha_v, device=device)
        A_t   = noise_attn(A0, alpha)
        logits = denoiser(X, A_t, alpha)
        pred   = logits.argmax(-1)           # [B,L]
        true_  = A0.argmax(-1)               # [B,L]
        # ignore padding (rows where A0 is all uniform → no valid argmax target)
        mask   = (A0.max(-1).values > 1.0/N_SENT + 0.01)  # [B,L]
        acc    = (pred == true_)[mask].float().mean().item() if mask.any() else float("nan")
        results[alpha_v] = acc
    return results

# ── Main ───────────────────────────────────────────────────────────────────
def run_seed(seed, train_data, val_data, test_data, tok):
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train_ds = SumDataset(train_data, tok)
    val_ds   = SumDataset(val_data,   tok)
    test_ds  = SumDataset(test_data,  tok)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, collate_fn=collate)

    enc       = SentEncoder(len(tok.vocab), D_MODEL).to(DEVICE)
    free_dec  = FreeDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    denoiser  = GNNDenoiser(D_MODEL, N_SENT, L_SUM, N_MP).to(DEVICE)
    guided_dec = GuidedDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    n_params = {
        "enc":     sum(p.numel() for p in enc.parameters()),
        "free":    sum(p.numel() for p in free_dec.parameters()),
        "denoiser":sum(p.numel() for p in denoiser.parameters()),
        "guided":  sum(p.numel() for p in guided_dec.parameters()),
    }
    print("  Params:", {k: f"{v:,}" for k, v in n_params.items()})

    # Phase 1: train free decoder
    print("  Phase 1: training free decoder...")
    train_free(enc, free_dec, train_loader, EPOCHS_FREE, DEVICE)

    # Extract A_0 from all data (train+val combined for denoiser training)
    print("  Extracting A_0...")
    all_loader = DataLoader(
        SumDataset(train_data + val_data, tok),
        batch_size=BATCH, shuffle=False, collate_fn=collate
    )
    A0_all, X_all, _ = extract_A0(enc, free_dec, all_loader, DEVICE)

    # Also extract sum_ids for guided decoder training
    all_sum_ids = torch.stack([
        torch.tensor(tok.enc(r["summary"] if "summary" in r else r["highlights"].split("\n")[0],
                              L_SUM, bos=True, eos=True), dtype=torch.long)
        for r in train_data + val_data
    ])
    A0_ds = A0Dataset(A0_all, X_all, all_sum_ids)

    # Phase 2: train denoiser
    print("  Phase 2: training denoiser...")
    train_denoiser(denoiser, A0_ds, EPOCHS_DEN, DEVICE)

    # Phase 3: train guided decoder
    print("  Phase 3: training guided decoder...")
    train_guided(enc, guided_dec, A0_ds, tok, EPOCHS_GUIDED, DEVICE)

    # Evaluate
    print("  Evaluating...")
    scores = evaluate(enc, free_dec, guided_dec, denoiser, test_loader, tok, DEVICE)

    # Recovery test
    A0_test, X_test, _ = extract_A0(enc, free_dec,
        DataLoader(test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate), DEVICE)
    rec = recovery_test(denoiser, A0_test, X_test)

    print("\n  ROUGE-1 F1 on test:")
    print(f"    free              : {scores['free']:.4f}")
    print(f"    oracle (true A_0) : {scores['oracle']:.4f}")
    for k in K_LIST:
        print(f"    guided-K{k}        : {scores[f'K{k}']:.4f}")

    print("\n  Recovery accuracy vs alpha:")
    print("   alpha   acc")
    for a, acc in sorted(rec.items()):
        print(f"   {a:.2f}   {acc:.3f}")

    return scores, rec

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    train_data, val_data, test_data = load_data()

    # Build vocab from train
    print("Building vocabulary...")
    tok = Tokenizer()
    all_texts = []
    for r in train_data + val_data:
        all_texts.append(r["document"] if "document" in r else r["article"])
        all_texts.append(r["summary"]  if "summary"  in r else r["highlights"])
    tok.build(all_texts)
    print(f"  Vocab size: {len(tok.vocab)}")

    all_scores = []
    all_recs   = []
    for seed in SEEDS:
        scores, rec = run_seed(seed, train_data, val_data, test_data, tok)
        all_scores.append(scores)
        all_recs.append(rec)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — mean over {len(SEEDS)} seeds")
    print(f"{'='*60}")

    keys = ["free", "oracle"] + [f"K{k}" for k in K_LIST]
    print(f"\n  ROUGE-1 F1:")
    for key in keys:
        mean = float(np.mean([s[key] for s in all_scores]))
        std  = float(np.std( [s[key] for s in all_scores]))
        delta = mean - np.mean([s["free"] for s in all_scores])
        tag = ""
        if key == "oracle":   tag = f"  Δfree={delta:+.4f}"
        elif key.startswith("K"): tag = f"  Δfree={delta:+.4f}"
        print(f"    {key:18s}: {mean:.4f} ± {std:.4f}{tag}")

    print("\n  Recovery accuracy (mean):")
    alphas = sorted(all_recs[0].keys())
    print("   alpha   acc")
    for a in alphas:
        mean_acc = float(np.mean([r[a] for r in all_recs if not math.isnan(r[a])]))
        print(f"   {a:.2f}   {mean_acc:.3f}")

    out = {
        "config": dict(D_MODEL=D_MODEL, N_HEADS=N_HEADS, N_LAYERS=N_LAYERS,
                       N_SENT=N_SENT, L_SUM=L_SUM, VOCAB_SIZE=VOCAB_SIZE,
                       BATCH=BATCH, EPOCHS_FREE=EPOCHS_FREE, EPOCHS_DEN=EPOCHS_DEN,
                       EPOCHS_GUIDED=EPOCHS_GUIDED, N_MP=N_MP),
        "scores": all_scores,
        "recovery": [{str(k): v for k, v in r.items()} for r in all_recs],
        "mean_scores": {k: float(np.mean([s[k] for s in all_scores])) for k in keys},
    }
    path = os.path.join(RESULTS_DIR, "summarization_attn_diff.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nFull results → {path}")

if __name__ == "__main__":
    main()
