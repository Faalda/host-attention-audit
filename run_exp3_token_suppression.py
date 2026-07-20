"""
Experiment 3 — Token-conditioned suppression.

Three suppression strategies compared on 500 test images:
  1. Random-K         : suppress K random patches (baseline)
  2. Global top-K     : current method — rank patches by sum_l A0[l,:]
  3. Token-conditioned: find the target noun's output position l* in the
                        caption, suppress top-K patches of A0[l*,:] only

Hypothesis: token-conditioned suppression raises primary-noun controllability
(strict metric) well above the current 28% global-top-K ceiling.

Metrics per strategy:
  - Primary-noun removal rate + 95% CI (Clopper-Pearson)
  - Non-target noun survival rate (selectivity)
  - Content preservation (ROUGE-1 suppressed vs. denoised)
  - Targeting precision (fraction of removed words that are nouns)

Also computes JS divergence between denoised and oracle A0 plans (Exp 2
partial — requires no additional training).

Output: results/token_suppression_eval.json
Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \
        run_exp3_token_suppression.py > results/token_supp.log 2>&1
"""

import os, sys, json, random, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import beta as beta_dist
import spacy
from rouge_score import rouge_scorer as rouge_mod

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import (
    Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS,
    L_SUM as L_CAP, BATCH, LR, N_MP,
)
from run_captioning import (
    PatchProjector, CaptionDecoder,
    CocoCapDataset, collate_cap, A0CapDataset,
    train_free_cap, extract_A0_cap, N_CAP_PER,
)
from run_cap_improved import (
    CFGDenoiser, SoftGuidedCaptionDecoder,
    ENCODER, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES,
    train_cfg_denoiser, train_guided_mixed, maskgit_cfg,
    RESULTS_DIR, CACHE_PATH,
)

N_EVAL   = 500    # test images to evaluate
SEED     = 42
TOP_K    = 3      # patches to suppress
EPOCHS_FREE  = 40
EPOCHS_DEN   = 60
EPOCHS_GUIDE = 50
ROUGE_THRESH = 0.95   # sim < THRESH → caption changed
NLP = spacy.load("en_core_web_sm")


def clopper_pearson(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha/2,   k,     n-k+1) if k > 0 else 0.0
    hi = beta_dist.ppf(1-alpha/2, k+1,   n-k)   if k < n else 1.0
    return float(lo), float(hi)


def rouge1(a, b):
    scorer = rouge_mod.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(a, b)['rouge1'].fmeasure


def extract_nouns_spacy(text):
    """Return list of (token_text, char_idx) for all NOUN/PROPN tokens."""
    doc = NLP(text.lower())
    return [(t.text, t.i) for t in doc if t.pos_ in ('NOUN', 'PROPN')]


def find_noun_token_positions(caption_toks, tok_obj):
    """
    Map spaCy noun spans in the decoded caption back to token indices in
    the captioner's vocabulary sequence.
    caption_toks: list[int] — token ids from tok_obj
    Returns: list of (noun_str, token_index) pairs
    """
    decoded = tok_obj.dec(caption_toks)
    nouns_spacy = extract_nouns_spacy(decoded)
    noun_strs = {n for n, _ in nouns_spacy}

    result = []
    for pos, tid in enumerate(caption_toks):
        word = (tok_obj.vocab[tid] if tid < len(tok_obj.vocab) else '').lower()
        if word in noun_strs:
            result.append((word, pos))
    return result


def js_divergence(p, q, eps=1e-8):
    """Jensen-Shannon divergence between two probability distributions."""
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log((p + eps) / (m + eps))).sum(-1)
    kl_qm = (q * torch.log((q + eps) / (m + eps))).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def mean_js_divergence(A0_denoised, A0_oracle):
    """
    A0_denoised, A0_oracle: [L, N] — softmax-normalised attention plans.
    Returns scalar mean JS divergence over all L positions.
    """
    # Normalise so each row sums to 1
    p = F.softmax(A0_denoised * 10, dim=-1)
    q = A0_oracle / A0_oracle.sum(-1, keepdim=True).clamp(1e-8)
    return js_divergence(p, q).mean().item()


@torch.no_grad()
def eval_three_strategies(proj, free_dec, guided_dec, denoiser,
                           test_feats, test_caps, tok, device, n_eval):
    """
    Evaluate all three suppression strategies on n_eval test images.
    Returns per-image records and aggregated stats.
    """
    proj.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()
    records = []
    js_divs = []   # JS divergence per image (Exp 2 partial)

    for img_i in range(min(n_eval, len(test_feats))):
        patches = test_feats[img_i].unsqueeze(0).to(device)
        X = proj(patches)
        N = X.shape[1]
        ref = test_caps[img_i][0] if test_caps[img_i] else ""

        # Free caption + oracle A0 (last-layer attention)
        free_toks = free_dec.generate(X)
        free_cap  = tok.dec(free_toks[0].cpu().tolist())
        _, attn_w = free_dec(X, free_toks[:, :-1], ret_last_attn=True)
        L_actual  = attn_w.shape[1]
        pad       = torch.zeros(1, L_CAP - L_actual, N, device=device)
        A0_oracle = torch.cat([attn_w, pad], dim=1).detach()[0]  # [L, N]

        # Denoised plan
        A_den    = maskgit_cfg(denoiser, X, n_steps=4, device=device)
        A0_denoised = F.softmax(A_den[0] * 10, dim=-1)   # [L, N]
        plan_den = A0_denoised.unsqueeze(0)
        den_cap  = tok.dec(guided_dec.generate(X, plan_den)[0].cpu().tolist())

        # JS divergence (denoised vs oracle)
        js_divs.append(mean_js_divergence(A0_denoised, A0_oracle))

        # Identify target noun (primary — first noun in free caption)
        nouns = extract_nouns_spacy(free_cap)
        primary_noun = nouns[0][0] if nouns else None
        all_nouns    = {n for n, _ in nouns}

        # ── Strategy 1: Random-K ────────────────────────────────────────────
        torch.manual_seed(img_i * 7 + SEED)
        rand_top = torch.randperm(N, device=device)[:TOP_K]

        # ── Strategy 2: Global top-K ────────────────────────────────────────
        global_mass  = A0_oracle.sum(0)       # [N]
        global_top   = global_mass.topk(TOP_K).indices

        # ── Strategy 3: Token-conditioned ──────────────────────────────────
        # Find the noun token positions in the caption
        noun_positions = find_noun_token_positions(
            free_toks[0].cpu().tolist(), tok)

        if noun_positions:
            # Use primary noun's first occurrence in token sequence
            primary_pos = noun_positions[0][1]
            primary_pos = min(primary_pos, L_CAP - 1)
            token_row   = A0_oracle[primary_pos]   # [N] attention of noun position
            token_top   = token_row.topk(TOP_K).indices
        else:
            # Fallback to global if no nouns found
            token_top = global_top

        captions = {}
        for strat_name, top_idxs in [
                ('random',  rand_top),
                ('global',  global_top),
                ('token',   token_top)]:
            A_sup = A0_oracle.clone()
            A_sup[:, top_idxs] = 0
            A_sup = A_sup / A_sup.sum(-1, keepdim=True).clamp(1e-9)
            plan_sup = A_sup.unsqueeze(0)
            sup_cap = tok.dec(guided_dec.generate(X, plan_sup)[0].cpu().tolist())
            captions[strat_name] = sup_cap

        # Compute metrics for each strategy
        def metrics_for(sup_cap):
            supp_nouns = {n for n, _ in extract_nouns_spacy(sup_cap)}
            pnoun_removed  = (primary_noun is not None) and (primary_noun not in supp_nouns)
            # Non-target noun survival: non-primary nouns that survive
            non_target = all_nouns - {primary_noun} if primary_noun else all_nouns
            surv_rate  = (len(non_target & supp_nouns) / len(non_target)
                         if non_target else 1.0)
            content_pres = rouge1(den_cap, sup_cap)
            # Targeting precision: fraction of removed words that are nouns
            from nltk import pos_tag, word_tokenize
            from nltk.corpus import stopwords
            STOP = set(stopwords.words("english"))
            NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
            free_tks = set(word_tokenize(free_cap.lower())) - STOP
            supp_tks = set(word_tokenize(sup_cap.lower())) - STOP
            removed  = free_tks - supp_tks
            if removed:
                tprec = sum(1 for w in removed if pos_tag([w])[0][1] in NOUN_TAGS) / len(removed)
            else:
                tprec = 1.0
            any_noun_removed = len(all_nouns - supp_nouns) > 0
            return {
                "pnoun_removed":    bool(pnoun_removed),
                "any_noun_removed": bool(any_noun_removed),
                "non_target_surv":  float(surv_rate),
                "content_pres":     float(content_pres),
                "targeting_prec":   float(tprec),
                "changed":          rouge1(free_cap, sup_cap) < ROUGE_THRESH,
            }

        rec = {
            "img_i":         img_i,
            "free_cap":      free_cap,
            "den_cap":       den_cap,
            "primary_noun":  primary_noun,
            "js_div":        js_divs[-1],
        }
        for strat in ('random', 'global', 'token'):
            rec[strat] = {
                "sup_cap": captions[strat],
                **metrics_for(captions[strat]),
            }
        records.append(rec)

        if img_i % 50 == 0:
            # Print running stats
            n_done = img_i + 1
            gp = sum(r['global']['pnoun_removed'] for r in records)
            tp = sum(r['token']['pnoun_removed']  for r in records)
            print(f"  [{n_done:4d}/{min(n_eval,len(test_feats))}]  "
                  f"global={gp/n_done:.1%}  token={tp/n_done:.1%}  "
                  f"mean_js={np.mean(js_divs):.4f}")

    return records, float(np.mean(js_divs))


def aggregate(records, strat):
    n   = len(records)
    k_p = sum(r[strat]['pnoun_removed']   for r in records)
    k_a = sum(r[strat].get('any_noun_removed', False) for r in records)
    k_s = sum(r[strat]['non_target_surv'] > 0.5 for r in records)   # >50% non-target survive
    ci_p_lo, ci_p_hi = clopper_pearson(k_p, n)
    ci_a_lo, ci_a_hi = clopper_pearson(k_a, n)
    ci_s_lo, ci_s_hi = clopper_pearson(k_s, n)
    return {
        "n":                     n,
        "any_noun_removed":      round(k_a / n, 3),
        "any_noun_k":            k_a,
        "any_ci_lo":             round(ci_a_lo, 3),
        "any_ci_hi":             round(ci_a_hi, 3),
        "primary_noun_removed":  round(k_p / n, 3),
        "primary_noun_k":        k_p,
        "ci_lo":                 round(ci_p_lo, 3),
        "ci_hi":                 round(ci_p_hi, 3),
        "selectivity_rate":      round(k_s / n, 3),
        "selectivity_k":         k_s,
        "sel_ci_lo":             round(ci_s_lo, 3),
        "sel_ci_hi":             round(ci_s_hi, 3),
        "mean_content_pres":     round(np.mean([r[strat]['content_pres']    for r in records]), 3),
        "mean_targeting_prec":   round(np.mean([r[strat]['targeting_prec']  for r in records]), 3),
        "change_rate":           round(np.mean([r[strat]['changed']         for r in records]), 3),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print(f"Loading features from {CACHE_PATH}...")
    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    total       = N_IMAGES + VAL_IMAGES + N_EVAL
    patch_feats = patch_feats[:total]
    print(f"  Loaded: {patch_feats.shape}")

    from datasets import load_dataset
    print("Loading MSCOCO captions...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    caps_per_img = []
    for item in ds.select(range(total)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    tok = Tokenizer()
    tok.build([c for caps in caps_per_img for c in caps], max_vocab=6000)
    print(f"  Vocab: {len(tok.vocab)}")

    train_feats = patch_feats[:N_IMAGES]
    tv_feats    = patch_feats[:N_IMAGES + VAL_IMAGES]
    test_feats  = patch_feats[N_IMAGES + VAL_IMAGES:]
    train_caps  = caps_per_img[:N_IMAGES]
    tv_caps     = caps_per_img[:N_IMAGES + VAL_IMAGES]
    test_caps   = caps_per_img[N_IMAGES + VAL_IMAGES:]

    proj       = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec   = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS,
                                n_src=N_PATCHES).to(DEVICE)
    denoiser   = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    guided_dec = SoftGuidedCaptionDecoder(
                     len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    train_ds     = CocoCapDataset(train_feats, train_caps, tok, proj)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_cap)

    print(f"  Phase 1: free decoder ({EPOCHS_FREE} epochs)...")
    train_free_cap(proj, free_dec, train_loader, EPOCHS_FREE, DEVICE)

    tv_ds     = CocoCapDataset(tv_feats, tv_caps, tok, proj)
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_all, X_all, _, ids_all = extract_A0_cap(proj, free_dec, tv_loader, DEVICE)
    A0_ds = A0CapDataset(A0_all, X_all, ids_all)

    print(f"  Phase 2: CFG denoiser ({EPOCHS_DEN} epochs)...")
    train_cfg_denoiser(denoiser, A0_ds, EPOCHS_DEN, DEVICE)

    print(f"  Phase 3: guided decoder ({EPOCHS_GUIDE} epochs)...")
    train_guided_mixed(guided_dec, A0_ds, denoiser, EPOCHS_GUIDE, DEVICE)

    print(f"\n  Evaluating {N_EVAL} images, three suppression strategies...")
    records, mean_js = eval_three_strategies(
        proj, free_dec, guided_dec, denoiser,
        test_feats, test_caps, tok, DEVICE, N_EVAL)

    # ── Aggregate stats ──────────────────────────────────────────────────────
    agg = {strat: aggregate(records, strat) for strat in ('random', 'global', 'token')}

    print("\n" + "="*60)
    print("Results (primary-noun removal rate, strict):")
    for strat in ('random', 'global', 'token'):
        a = agg[strat]
        print(f"  {strat:8s}: {a['primary_noun_removed']:.1%}  "
              f"({a['primary_noun_k']}/{a['n']})  "
              f"CI [{a['ci_lo']:.1%}, {a['ci_hi']:.1%}]  "
              f"sel={a['selectivity_rate']:.1%}  "
              f"pres={a['mean_content_pres']:.3f}")
    print(f"\n  Mean JS divergence (denoised vs oracle): {mean_js:.4f}")
    print("="*60)

    out = {
        "n_eval":    N_EVAL,
        "top_k":     TOP_K,
        "seed":      SEED,
        "mean_js_divergence_denoised_vs_oracle": round(mean_js, 5),
        "strategies": agg,
        "records":   [
            {k: v for k, v in r.items() if k != 'random'} for r in records
        ],   # omit random per-record to keep file small
    }
    path = os.path.join(RESULTS_DIR, "token_suppression_eval.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
