"""
Phase III — Residual nonlinear editor for concept substitution in A₀ plan space.
Trains a small MLP:  A' = A + f(A, e_src, e_tgt)
If mean target-noun rate >= 20%: linear arithmetic was the bottleneck (PASS).
If < 20%: flow matching unlikely to help either (KILL).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \
        run_residual_editor.py > results/residual_editor.log 2>&1
Prerequisite: run_save_plans.py must have completed.
"""

import os, sys, json, re, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, N_MP
from run_captioning import PatchProjector, CaptionDecoder
from run_cap_improved import (ENCODER, N_PATCHES, ENCODER_DIM, RESULTS_DIR,
                               SoftGuidedCaptionDecoder, CFGDenoiser)

SAVE_DIR   = os.path.join(RESULTS_DIR, "saved_plans")
SEED       = 42
NOUN_DIM   = 32
HIDDEN     = 256
EPOCHS     = 50
BATCH      = 32
LR         = 3e-4
N_TEST     = 30
PASS_THRESH = 0.20
ARITH_MEAN  = 0.038   # known baseline from plan_arithmetic.json

EDIT_PAIRS = [
    ("dog",    "cat"),
    ("chair",  "couch"),
    ("train",  "car"),
    ("bird",   "airplane"),
    ("person", "bicycle"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def caption_contains(caption, noun):
    c = caption.lower()
    return any(v in c.split() for v in [noun, noun + "s", noun + "es",
                                         noun.rstrip("e") + "ing"])


# ── Model ─────────────────────────────────────────────────────────────────────

class ResidualEditor(nn.Module):
    def __init__(self, n_patches, noun_dim, hidden, n_nouns):
        super().__init__()
        self.noun_emb = nn.Embedding(n_nouns, noun_dim)
        self.net = nn.Sequential(
            nn.Linear(n_patches + 2 * noun_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_patches),
        )

    def forward(self, A, src_idx, tgt_idx):
        # A: [B, L, N],  src_idx / tgt_idx: [B]
        e_src = self.noun_emb(src_idx).unsqueeze(1).expand(-1, A.shape[1], -1)
        e_tgt = self.noun_emb(tgt_idx).unsqueeze(1).expand(-1, A.shape[1], -1)
        x = torch.cat([A, e_src, e_tgt], dim=-1)           # [B, L, N+2D]
        delta = self.net(x)                                  # [B, L, N]
        A_prime = F.relu(A + delta)
        A_prime = A_prime / (A_prime.sum(-1, keepdim=True) + 1e-8)
        return A_prime


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_plans(split="tv"):
    plans = torch.load(os.path.join(SAVE_DIR, f"plans_{split}.pt"),
                       weights_only=True)
    with open(os.path.join(SAVE_DIR, f"noun_labels_{split}.json")) as f:
        labels = json.load(f)
    return plans, labels


def group_by_noun(plans, labels):
    groups = {}
    for i, noun in enumerate(labels):
        if noun is None:
            continue
        groups.setdefault(noun, []).append(i)
    return groups


# ── Training ──────────────────────────────────────────────────────────────────

def build_noun_vocab(groups):
    nouns = sorted(groups.keys())
    noun2idx = {n: i for i, n in enumerate(nouns)}
    return nouns, noun2idx


def train_editor(editor, plans_tv, groups_tv, noun2idx, device):
    opt = torch.optim.Adam(editor.parameters(), lr=LR)
    editor.train()
    valid_pairs = [(s, t) for s, t in EDIT_PAIRS
                   if s in groups_tv and t in groups_tv]
    print(f"  Training on {len(valid_pairs)} valid edit pairs:")
    for s, t in valid_pairs:
        print(f"    {s}→{t}: {len(groups_tv[s])} src, {len(groups_tv[t])} tgt plans")

    for epoch in range(1, EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for src_noun, tgt_noun in valid_pairs:
            src_idxs = groups_tv[src_noun]
            tgt_idxs = groups_tv[tgt_noun]
            n_samples = min(200, len(src_idxs))
            chosen_src = random.sample(src_idxs, n_samples)
            chosen_tgt = [random.choice(tgt_idxs) for _ in chosen_src]

            for b_start in range(0, n_samples, BATCH):
                b_src = chosen_src[b_start: b_start + BATCH]
                b_tgt = chosen_tgt[b_start: b_start + BATCH]
                A_src = plans_tv[b_src].to(device)          # [B, L, N]
                A_tgt = plans_tv[b_tgt].to(device)
                si = torch.full((len(b_src),), noun2idx[src_noun],
                                dtype=torch.long, device=device)
                ti = torch.full((len(b_src),), noun2idx[tgt_noun],
                                dtype=torch.long, device=device)
                A_prime = editor(A_src, si, ti)
                loss = F.mse_loss(A_prime, A_tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); n_batches += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/max(n_batches,1):.5f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(editor, plans_test, labels_test, noun2idx, proj, guided_dec, tok,
             device):
    editor.eval(); proj.eval(); guided_dec.eval()

    # We need image features for the test split to produce X for the guided decoder.
    # We don't have them at this stage — guided_dec.generate(X, attn_plan) needs X.
    # Workaround: use the PLANS themselves as a plan-only evaluation.
    # Measure: after editing A_src → A_prime, does the *peak patch* shift toward
    # patches associated with the target noun?
    # More meaningful: decode captions using guided_dec with A_prime as attn_plan
    # but zero X (pure plan-driven generation), which tests whether the plan alone
    # guides the noun.
    #
    # Actually: load the feature cache to get real image features for the test set.
    from run_cap_improved import CACHE_PATH, N_IMAGES, VAL_IMAGES, TEST_IMAGES
    if not os.path.exists(CACHE_PATH):
        print("  WARNING: feature cache not found; skipping feature-based eval")
        return {}

    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    n_tv = N_IMAGES + VAL_IMAGES
    test_feats = patch_feats[n_tv: n_tv + TEST_IMAGES]   # [1000, N, D_enc]

    results = {}
    for src_noun, tgt_noun in EDIT_PAIRS:
        if src_noun not in noun2idx or tgt_noun not in noun2idx:
            print(f"  Skipping {src_noun}→{tgt_noun} (not in noun vocab)")
            continue

        # find test images with src_noun
        src_test_idxs = [i for i, n in enumerate(labels_test)
                         if n == src_noun][:N_TEST]
        if not src_test_idxs:
            print(f"  No test images for {src_noun}, skipping")
            continue

        si = torch.full((len(src_test_idxs),), noun2idx[src_noun],
                        dtype=torch.long, device=device)
        ti = torch.full((len(src_test_idxs),), noun2idx[tgt_noun],
                        dtype=torch.long, device=device)
        A_src = plans_test[src_test_idxs].to(device)
        A_prime = editor(A_src, si, ti)

        # get projected features for these test images
        feats = test_feats[src_test_idxs].to(device)       # [n, N_patches, D_enc]
        X = proj(feats)                                      # [n, N_patches, D_model]

        # decode using guided decoder with the edited plan as attention prior
        ids = guided_dec.generate(X, A_prime)               # [n, L]
        caps = [tok.dec(row.cpu().tolist()) for row in ids]

        n_hit = sum(1 for c in caps if caption_contains(c, tgt_noun))
        rate  = n_hit / len(caps)
        delta = rate - ARITH_MEAN
        print(f"  {src_noun}→{tgt_noun}: {n_hit}/{len(caps)} = {rate:.3f}  "
              f"(arith={ARITH_MEAN:.3f}, Δ={delta:+.3f})")
        for cap in caps[:3]:
            print(f"    sample: {cap}")

        results[f"{src_noun}→{tgt_noun}"] = {
            "n_test":          len(caps),
            "target_appeared": round(rate, 4),
            "vs_arithmetic":   f"{delta:+.4f}",
        }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); torch.manual_seed(SEED)

    # ── Load plans ────────────────────────────────────────────────────────────
    print("Loading plans...")
    plans_tv,   labels_tv   = load_plans("tv")
    plans_test, labels_test = load_plans("test")
    print(f"  tv:   {plans_tv.shape}, test: {plans_test.shape}")

    groups_tv = group_by_noun(plans_tv, labels_tv)
    nouns, noun2idx = build_noun_vocab(groups_tv)
    print(f"  Noun vocab ({len(nouns)}): {nouns[:10]}...")

    # ── Build editor ──────────────────────────────────────────────────────────
    editor = ResidualEditor(N_PATCHES, NOUN_DIM, HIDDEN, len(nouns)).to(DEVICE)
    n_params = sum(p.numel() for p in editor.parameters())
    print(f"  ResidualEditor: {n_params:,} parameters")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining residual editor...")
    train_editor(editor, plans_tv, groups_tv, noun2idx, DEVICE)

    # ── Load decoder models ───────────────────────────────────────────────────
    print("\nLoading decoder models...")
    with open(os.path.join(SAVE_DIR, "tok_vocab.json")) as f:
        vocab = json.load(f)
    tok = Tokenizer(); tok.vocab = vocab
    tok.w2i = {w: i for i, w in enumerate(vocab)}

    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"),
                                    weights_only=True, map_location=DEVICE))

    guided_dec = SoftGuidedCaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    guided_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"),
                                          weights_only=True, map_location=DEVICE))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating edited captions...")
    results = evaluate(editor, plans_test, labels_test, noun2idx,
                       proj, guided_dec, tok, DEVICE)

    if not results:
        print("No results produced.")
        return

    mean_rate = sum(v["target_appeared"] for v in results.values()) / len(results)
    improvement = mean_rate - ARITH_MEAN
    verdict = "PASS" if mean_rate >= PASS_THRESH else "KILL"
    print(f"\n{'='*60}")
    print(f"RESIDUAL EDITOR SUMMARY  ({len(results)} pairs)")
    print(f"  Mean target-appeared: {mean_rate:.4f}")
    print(f"  Arithmetic baseline:  {ARITH_MEAN:.4f}")
    print(f"  Improvement:          {improvement:+.4f}")
    print(f"  Verdict:              {verdict}")
    if verdict == "PASS":
        print("  → Nonlinear editing signal confirmed; flow matching is justified.")
    else:
        print("  → No improvement over arithmetic; plan space may not encode concept.")
    print(f"{'='*60}")

    out = {
        "noun_pairs":  results,
        "summary": {
            "mean_target_appeared":      round(mean_rate, 4),
            "arithmetic_mean":           ARITH_MEAN,
            "improvement_over_arithmetic": round(improvement, 4),
            "verdict":                   verdict,
        },
    }
    path = os.path.join(RESULTS_DIR, "residual_editor.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
