"""
Masked Attention Diffusion for summarization.

Fix for "denoiser ignores A_t": replace continuous noise with row masking.

Noise model:
  A_t[i,:] = A_0[i,:]   if row i is UNMASKED  (exact, informative)
  A_t[i,:] = uniform     if row i is MASKED    (to predict)

This forces cross-row GNN propagation: unmasked rows reveal the attention
pattern for neighboring positions. The denoiser MUST use A_t (unmasked rows)
because X alone can't distinguish adjacent sentence-level patterns.

Inference: MaskGIT-style iterative unmasking — each step adds context for next.

Key diagnostic: recovery accuracy on masked rows at mask_rate=0.5 should be
>> 39.1% (the X-only baseline from continuous noise).
"""

import os, sys, json, re, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import (
    Tokenizer, SumDataset, collate, SentEncoder, FreeDecoder,
    GuidedDecoder, train_free, extract_A0, A0Dataset, train_guided,
    DEVICE, D_MODEL, N_HEADS, N_LAYERS, N_SENT, SENT_LEN,
    L_SUM, VOCAB_SIZE, BATCH, LR, EPOCHS_FREE, EPOCHS_GUIDED, N_MP,
    split_sentences, multi_step_denoise, noise_attn, eval_rouge,
)

EPOCHS_MASK_DEN = 50
MASK_RATES      = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
K_LIST          = [1, 2, 4, 8]
SEEDS           = [42, 7, 13]
RESULTS_DIR     = "/home/faldabbas/myproject/attn_diffusion_gen/results"

# ── Masked noise ────────────────────────────────────────────────────────────
def mask_attn(A0, mask_rate):
    """
    Mask each row independently with probability mask_rate.
    Returns (A_t, mask) where mask=1 means the row was masked (uniform).
    """
    B, L, N = A0.shape
    mask = (torch.rand(B, L, device=A0.device) < mask_rate).float()  # [B,L]
    uniform = torch.full_like(A0, 1.0 / N)
    A_t = mask.unsqueeze(-1) * uniform + (1 - mask.unsqueeze(-1)) * A0
    return A_t, mask

# ── Masked GNN Denoiser ─────────────────────────────────────────────────────
class MaskedGNNDenoiser(nn.Module):
    """
    GNN denoiser with explicit mask embedding.
    Unmasked rows in A_t carry exact information.
    Masked rows must be predicted from X + neighboring unmasked rows.
    """
    def __init__(self, D, n_src=N_SENT, n_tgt=L_SUM, n_mp=N_MP):
        super().__init__()
        self.row_proj   = nn.Linear(n_src, D)
        self.mask_emb   = nn.Embedding(2, D)        # 0=known, 1=masked
        self.msg_projs  = nn.ModuleList([nn.Linear(D, D) for _ in range(n_mp)])
        self.mp_norms   = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_mp)])
        # Position-aware self-attention over target positions
        self.pos_attn   = nn.MultiheadAttention(D, 4, batch_first=True)
        self.pos_norm   = nn.LayerNorm(D)
        self.q_proj     = nn.Linear(D, D)
        self.k_proj     = nn.Linear(D, D)
        self.scale      = D ** -0.5

    def forward(self, X_src, A_t, mask):
        # X_src: [B,N,D]   A_t: [B,L,N]   mask: [B,L] int 0/1
        B, L, N = A_t.shape
        h = F.gelu(self.row_proj(A_t))              # [B,L,D]
        h = h + self.mask_emb(mask.long())           # add mask type embedding

        # GNN: message-pass from unmasked rows (carry real A_t) to masked
        for mp, ln in zip(self.msg_projs, self.mp_norms):
            # Use full A_t (unmasked rows have clean values, masked have uniform)
            msgs = torch.bmm(A_t, mp(X_src))         # [B,L,D]
            h    = ln(h + msgs)

        # Self-attention over positions: propagate unmasked context to masked
        # Weight unmasked rows more heavily by masking them out in key/value
        known_mask = (mask == 0)                      # [B,L] True = known/unmasked
        # attn_bias: penalise attending to masked positions from masked positions
        bias = torch.zeros(B * 4, L, L, device=A_t.device)
        h2, _ = self.pos_attn(h, h, h, need_weights=False)
        h = self.pos_norm(h + h2)

        Q = F.normalize(self.q_proj(h), dim=-1)       # [B,L,D]
        K = self.k_proj(X_src)                        # [B,N,D]
        return torch.bmm(Q, K.transpose(-1,-2)) / self.scale   # [B,L,N]

# ── Training ─────────────────────────────────────────────────────────────────
def train_masked_denoiser(denoiser, A0_dataset, n_epochs, device):
    loader = DataLoader(A0_dataset, batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(denoiser.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0; n = 0
        for A0, X, _ in loader:
            A0 = A0.to(device); X = X.to(device)
            B = A0.shape[0]
            # Sample mask rate uniformly in [0.1, 0.95]
            mask_rate = random.uniform(0.1, 0.95)
            A_t, mask = mask_attn(A0, mask_rate)
            logits = denoiser(X, A_t, mask)           # [B,L,N]
            # Loss only on MASKED rows
            tgt = A0.argmax(-1)                        # [B,L]
            # Select only masked positions
            m = mask.bool()                            # [B,L]
            if m.sum() == 0: continue
            loss = F.cross_entropy(
                logits[m],    # [M, N_SENT]
                tgt[m]        # [M]
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  mask_den_loss={tot/n:.4f}")

# ── MaskGIT inference ────────────────────────────────────────────────────────
@torch.no_grad()
def maskgit_denoise(denoiser, X_src, n_steps, device=None):
    """
    Iterative unmasking: each step unmasks the most-confident 1/n_steps fraction.
    Step 1: all masked → predict from X alone (like continuous baseline)
    Step K: most unmasked → predict remaining from rich context
    """
    if device is None: device = X_src.device
    B, N, D = X_src.shape
    # Start: all masked → A_t = uniform
    A = torch.full((B, L_SUM, N), 1.0 / N, device=device)
    mask = torch.ones(B, L_SUM, device=device)          # all masked
    unmasked_so_far = torch.zeros(B, L_SUM, dtype=torch.bool, device=device)

    n_unmask_per_step = max(1, L_SUM // n_steps)

    for step in range(n_steps):
        # Predict
        logits = denoiser(X_src, A, mask)               # [B,L,N]
        probs  = F.softmax(logits, dim=-1)               # [B,L,N]
        conf   = probs.max(-1).values                    # [B,L] confidence

        # Only consider still-masked positions
        conf_masked = conf.masked_fill(~mask.bool(), -1e9)

        # Unmask top n_unmask_per_step positions per batch item
        _, topk_idx = conf_masked.topk(
            min(n_unmask_per_step, (mask > 0).sum(-1).min().item()),
            dim=-1
        )  # [B, k]

        # Commit predicted values at unmasked positions
        pred_hard = probs.argmax(-1)                     # [B,L]
        for b in range(B):
            for idx in topk_idx[b]:
                if mask[b, idx] > 0:
                    # One-hot the predicted sentence
                    A[b, idx] = 0
                    A[b, idx, pred_hard[b, idx]] = 1.0
                    mask[b, idx] = 0

    # Final pass: use full unmasked A to refine remaining
    final_logits = denoiser(X_src, A, mask)
    A_out = F.softmax(final_logits, dim=-1)
    # Blend with committed values
    committed = (mask == 0).unsqueeze(-1).float()
    return committed * A + (1 - committed) * A_out      # [B,L,N]

# ── Recovery test ────────────────────────────────────────────────────────────
@torch.no_grad()
def recovery_test(denoiser, A0_tensor, X_tensor, n_samples=256, device=DEVICE):
    denoiser.eval()
    idx = torch.randperm(len(A0_tensor))[:n_samples]
    A0  = A0_tensor[idx].to(device)
    X   = X_tensor[idx].to(device)
    results = {}
    for mr in MASK_RATES:
        A_t, mask = mask_attn(A0, mr)
        logits = denoiser(X, A_t, mask)
        pred   = logits.argmax(-1)
        tgt    = A0.argmax(-1)
        # Measure accuracy on MASKED rows only (where the model had to predict)
        m      = mask.bool()
        if not m.any():
            results[mr] = float('nan')
            continue
        # Also compute: accuracy on unmasked rows (should be ~100% → model reads them)
        acc_masked = (pred == tgt)[m].float().mean().item()
        results[mr] = acc_masked
    return results

# ── Evaluation ────────────────────────────────────────────────────────────────
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

        # Free
        toks = free_dec.generate(X)
        free_preds += [tok.dec(r) for r in toks.cpu().tolist()]

        # Oracle A_0
        _, attn_w = free_dec(X, s[:, :-1], ret_last_attn=True)
        pad = torch.zeros(B, L_SUM - attn_w.shape[1], N_SENT, device=device)
        A_oracle = torch.cat([attn_w, pad], dim=1)
        plan_or  = F.softmax(A_oracle.detach() * 10, dim=-1)
        oracle_preds += [tok.dec(r) for r in guided_dec.generate(X, plan_or).cpu().tolist()]
        refs += batch["ref"]

        # MaskGIT K steps
        for k in K_LIST:
            A_mg = maskgit_denoise(denoiser, X, n_steps=k, device=device)
            plan_mg = F.softmax(A_mg * 10, dim=-1)
            k_preds[k] += [tok.dec(r) for r in guided_dec.generate(X, plan_mg).cpu().tolist()]

    return {
        "free":   eval_rouge(free_preds,   refs),
        "oracle": eval_rouge(oracle_preds, refs),
        **{f"K{k}": eval_rouge(k_preds[k], refs) for k in K_LIST},
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def run_seed(seed, train_data, val_data, test_data, tok):
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train_ds = SumDataset(train_data, tok)
    val_ds   = SumDataset(val_data,   tok)
    test_ds  = SumDataset(test_data,  tok)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, collate_fn=collate)

    enc        = SentEncoder(len(tok.vocab), D_MODEL).to(DEVICE)
    free_dec   = FreeDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    mask_den   = MaskedGNNDenoiser(D_MODEL, N_SENT, L_SUM, N_MP).to(DEVICE)
    guided_dec = GuidedDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    total_den = sum(p.numel() for p in mask_den.parameters())
    print(f"  MaskedGNNDenoiser params: {total_den:,}")

    # Phase 1: train free decoder
    print("  Phase 1: training free decoder...")
    train_free(enc, free_dec, train_loader, EPOCHS_FREE, DEVICE)

    # Extract A_0
    print("  Extracting A_0...")
    all_loader = DataLoader(
        SumDataset(train_data + val_data, tok),
        batch_size=BATCH, shuffle=False, collate_fn=collate
    )
    A0_all, X_all, _ = extract_A0(enc, free_dec, all_loader, DEVICE)
    all_sum_ids = torch.stack([
        torch.tensor(tok.enc(
            r["summary"] if "summary" in r else r["highlights"].split("\n")[0],
            L_SUM, bos=True, eos=True
        ), dtype=torch.long)
        for r in train_data + val_data
    ])
    A0_ds = A0Dataset(A0_all, X_all, all_sum_ids)

    # Phase 2: train MASKED denoiser
    print("  Phase 2: training masked denoiser...")
    train_masked_denoiser(mask_den, A0_ds, EPOCHS_MASK_DEN, DEVICE)

    # Phase 3: train guided decoder
    print("  Phase 3: training guided decoder...")
    train_guided(enc, guided_dec, A0_ds, tok, EPOCHS_GUIDED, DEVICE)

    # Recovery test
    A0_test, X_test, _ = extract_A0(enc, free_dec,
        DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate), DEVICE)
    rec = recovery_test(mask_den, A0_test, X_test)

    # Evaluate ROUGE
    print("  Evaluating ROUGE...")
    scores = evaluate(enc, free_dec, guided_dec, mask_den, test_loader, tok, DEVICE)

    print("\n  ROUGE-1 F1:")
    for k, v in scores.items():
        delta = v - scores["free"]
        tag = f"  Δfree={delta:+.4f}" if k != "free" else ""
        print(f"    {k:20s}: {v:.4f}{tag}")

    print("\n  Recovery on masked rows (mask_rate → accuracy):")
    print("   mask_rate   acc_masked   vs_x_only(0.391)")
    for mr, acc in sorted(rec.items()):
        delta = acc - 0.391  # baseline from continuous noise
        print(f"   {mr:.2f}        {acc:.3f}        {delta:+.3f}")

    return scores, rec

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from datasets import load_dataset
    from run_summarization import TRAIN_N, VAL_N, TEST_N
    print("Loading XSum...")
    ds = load_dataset("xsum")
    train_data = list(ds["train"].select(range(TRAIN_N)))
    val_data   = list(ds["validation"].select(range(VAL_N)))
    test_data  = list(ds["test"].select(range(TEST_N)))

    print("Building vocabulary...")
    tok = Tokenizer()
    all_texts = []
    for r in train_data + val_data:
        all_texts.append(r["document"]); all_texts.append(r["summary"])
    tok.build(all_texts)

    all_scores, all_recs = [], []
    for seed in SEEDS:
        scores, rec = run_seed(seed, train_data, val_data, test_data, tok)
        all_scores.append(scores); all_recs.append(rec)

    print(f"\n{'='*60}")
    print(f"SUMMARY — mean over {len(SEEDS)} seeds")
    print(f"{'='*60}")

    keys = ["free", "oracle"] + [f"K{k}" for k in K_LIST]
    free_mean = float(np.mean([s["free"] for s in all_scores]))
    print("\n  ROUGE-1 F1:")
    for key in keys:
        m = float(np.mean([s[key] for s in all_scores]))
        std = float(np.std([s[key] for s in all_scores]))
        print(f"    {key:20s}: {m:.4f} ± {std:.4f}  Δfree={m-free_mean:+.4f}")

    print("\n  Recovery on masked rows (mean):")
    print("   mask_rate   acc_masked   vs_x_only")
    for mr in MASK_RATES:
        vals = [r[mr] for r in all_recs if not math.isnan(r[mr])]
        m = float(np.mean(vals)) if vals else float('nan')
        print(f"   {mr:.2f}        {m:.3f}        {m-0.391:+.3f}")

    out = {
        "config": {"noise": "masked", "epochs_den": EPOCHS_MASK_DEN},
        "scores": all_scores,
        "recovery": [{str(k): v for k, v in r.items()} for r in all_recs],
        "mean_scores": {k: float(np.mean([s[k] for s in all_scores])) for k in keys},
    }
    path = os.path.join(RESULTS_DIR, "sum_masked_attn_diff.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nFull results → {path}")

if __name__ == "__main__":
    main()
