"""
run_make_figures.py — Generate qualitative editing figures for the AAAI paper.

Loads trained checkpoints from results/saved_plans/ and MSCOCO test images,
produces two PDF figures:

  paper/figures/editing_examples.pdf   — 4-row × 4-col: image | heatmap |
                                          suppressed heatmap | captions
  paper/figures/transplant_example.pdf — 4-row × 3-col: source | donor |
                                          transplanted caption

Usage:
    python3 run_make_figures.py
Prerequisite: run_save_plans.py must have completed (model_*.pt must exist).
"""

import os, sys, json, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, N_MP
from run_captioning import PatchProjector, CaptionDecoder, N_CAP_PER
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    RESULTS_DIR, CACHE_PATH,
    SoftGuidedCaptionDecoder, CFGDenoiser, maskgit_cfg,
    CFG_SCALE,
)

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")

# Attention heatmap colormap: transparent → orange-red
ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attn", [(0, "#00000000"), (0.3, "#ff880055"), (1.0, "#cc2200cc")]
)


# ── Tokenizer restore ─────────────────────────────────────────────────────────

def load_tokenizer():
    vocab_path = os.path.join(SAVE_DIR, "tok_vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    tok = Tokenizer()
    tok.vocab = vocab
    tok.w2i = {w: i for i, w in enumerate(vocab)}
    return tok


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(tok):
    V = len(tok.vocab)
    proj       = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec   = CaptionDecoder(V, D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES).to(DEVICE)
    denoiser   = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    guided_dec = SoftGuidedCaptionDecoder(V, D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    proj.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "model_proj.pt"), map_location=DEVICE, weights_only=True))
    free_dec.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "model_free.pt"), map_location=DEVICE, weights_only=True))
    denoiser.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"), map_location=DEVICE, weights_only=True))
    guided_dec.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "model_guided.pt"), map_location=DEVICE, weights_only=True))

    proj.eval(); free_dec.eval(); denoiser.eval(); guided_dec.eval()
    print("  Models loaded OK")
    return proj, free_dec, denoiser, guided_dec


# ── Attention utilities ───────────────────────────────────────────────────────

def extract_oracle_plan(proj, free_dec, feat):
    """Extract oracle A₀ from the free decoder for one image."""
    X = proj(feat.unsqueeze(0).to(DEVICE))          # [1, N, D]
    free_toks = free_dec.generate(X)                 # [1, L]
    dummy_ids = free_toks[:, :-1]
    _, attn_w = free_dec(X, dummy_ids, ret_last_attn=True)   # [1, L-1, N]
    L1 = attn_w.shape[1]
    pad = torch.zeros(1, L_CAP - L1, N_PATCHES, device=DEVICE)
    A0 = torch.cat([attn_w, pad], dim=1).detach()   # [1, L_CAP, N]
    return X, A0                                     # both on DEVICE


def suppress_plan(A0, k=20):
    """Zero top-k attended patches; renormalise rows. A0: [1, L, N]."""
    importance = A0[0].sum(0)                        # [N]
    top_k = importance.topk(k).indices
    A_sup = A0.clone()
    A_sup[0, :, top_k] = 0.0
    row_sums = A_sup[0].sum(-1, keepdim=True).clamp(min=1e-9)
    A_sup[0] = A_sup[0] / row_sums
    return A_sup, top_k.cpu().tolist()


def plan_to_numpy_heatmap(A0, grid=16):
    """A0: [1, L, N] or [L, N] → mean over L → [grid, grid] float32 in [0,1]."""
    if A0.dim() == 3:
        A0 = A0[0]
    attn = A0.mean(0).cpu().float().numpy()          # [N]
    hm = attn.reshape(grid, grid)
    mn, mx = hm.min(), hm.max()
    if mx > mn:
        hm = (hm - mn) / (mx - mn)
    return hm


@torch.no_grad()
def decode_with_plan(guided_dec, X, A0, tok):
    """Greedy decode given X [1,N,D] and A0 [1,L,N]."""
    plan = F.softmax(A0 * 10, dim=-1)
    toks = guided_dec.generate(X, plan)              # [1, L]
    return tok.dec(toks[0].cpu().tolist())


# ── Figure helpers ────────────────────────────────────────────────────────────

def pil_resize(img, size=224):
    return img.resize((size, size), Image.LANCZOS)


def show_image(ax, img_pil, title=None):
    ax.imshow(np.array(pil_resize(img_pil)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)


def show_heatmap(ax, img_pil, hm_np, alpha=0.65, title=None):
    """Overlay attention heatmap on image."""
    ax.imshow(np.array(pil_resize(img_pil)))
    hm_up = np.array(Image.fromarray((hm_np * 255).astype(np.uint8)).resize(
        (224, 224), Image.BILINEAR)) / 255.0
    ax.imshow(hm_up, cmap=ATTN_CMAP, alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)


def show_patch_markers(ax, img_pil, patch_ids, color, alpha=0.60, grid=16, title=None):
    """Overlay filled circles on the given patch positions. No heatmap — just markers."""
    ax.imshow(np.array(pil_resize(img_pil)))
    cell = 224 / grid
    for pid in patch_ids:
        row_g, col_g = divmod(int(pid), grid)
        cx = (col_g + 0.5) * cell
        cy = (row_g + 0.5) * cell
        circle = plt.Circle((cx, cy), cell * 0.44,
                             color=color, fill=True, alpha=alpha, linewidth=0)
        ax.add_patch(circle)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)


def show_suppressed_patches(ax, img_pil, hm_np, suppressed_patch_ids, grid=16, title=None):
    """Show image with suppressed patches marked as red circles."""
    show_patch_markers(ax, img_pil, suppressed_patch_ids,
                       color="#dd0000", alpha=0.60, grid=grid, title=title)


def show_caption_panel(ax, cap_orig, cap_supp, title=None):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    # word-wrap at ~35 chars
    def wrap(s, w=35):
        words = s.split()
        lines, cur = [], []
        for word in words:
            if sum(len(c) + 1 for c in cur) + len(word) > w and cur:
                lines.append(" ".join(cur)); cur = []
            cur.append(word)
        if cur: lines.append(" ".join(cur))
        return "\n".join(lines)

    ax.text(0.04, 0.92, "Original:", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", va="top", color="#222222")
    ax.text(0.04, 0.80, wrap(cap_orig), transform=ax.transAxes,
            fontsize=7, va="top", color="#333333", linespacing=1.4)
    ax.text(0.04, 0.42, "Suppressed:", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", va="top", color="#cc3300")
    ax.text(0.04, 0.30, wrap(cap_supp), transform=ax.transAxes,
            fontsize=7, va="top", color="#cc3300", linespacing=1.4)


# ── Main figures ──────────────────────────────────────────────────────────────

def make_editing_figure(proj, free_dec, denoiser, guided_dec, tok,
                         test_feats, plans_test, test_imgs, noun_labels_test,
                         n_examples=4, max_scan=500):
    """
    Selects n_examples test images where patch suppression changes at least
    one content word, then renders a 4-row × 4-col figure:
      Col 0: source image
      Col 1: attention heatmap (hot patches = most attended)
      Col 2: same image with suppressed patches marked as red circles
      Col 3: original caption (black) + suppressed caption (red)

    Requires noun_label != None to prefer named-object examples.
    Scans up to max_scan images; falls back to any word-change if noun examples
    are not found.
    """
    candidates = []
    print(f"  Scanning up to {max_scan} test images for suppression examples...")

    for i in range(min(max_scan, len(test_feats))):
        feat = test_feats[i]

        with torch.no_grad():
            X, A0 = extract_oracle_plan(proj, free_dec, feat)

        A0_sup, sup_ids = suppress_plan(A0, k=20)

        cap_orig = decode_with_plan(guided_dec, X, A0, tok)
        cap_supp = decode_with_plan(guided_dec, X, A0_sup, tok)

        if cap_orig == cap_supp:
            continue
        # Skip garbled captions (UNK tokens or very short)
        if "<UNK>" in cap_orig or "<UNK>" in cap_supp or len(cap_orig.split()) < 5:
            continue

        hm_orig = plan_to_numpy_heatmap(A0)
        noun    = noun_labels_test[i]

        # Score: noun-removed examples get 2.0 bonus; then word-diff fraction
        word_diff = _word_diff(cap_orig, cap_supp)
        noun_removed = (noun is not None and noun not in cap_supp.lower())
        score = word_diff + (2.0 if noun_removed else 0.0)

        candidates.append({
            "img":       test_imgs[i],
            "cap_orig":  cap_orig,
            "cap_supp":  cap_supp,
            "hm_orig":   hm_orig,
            "sup_ids":   sup_ids,
            "noun":      noun,
            "score":     score,
        })
        print(f"    [cand] idx={i}  noun={noun}  score={score:.2f}")
        print(f"         orig: {cap_orig}")
        print(f"         supp: {cap_supp}")

    # Pick top-n by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    final = candidates[:n_examples]

    if not final:
        print("  WARNING: no suppression examples found in scanned range")
        return None

    n_rows = len(final)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, 3.2 * n_rows),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Attention map", "Suppressed patches\n(red = zeroed)", "Captions"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=4)

    for row, ex in enumerate(final):
        show_image(axes[row, 0], ex["img"])
        show_heatmap(axes[row, 1], ex["img"], ex["hm_orig"])
        show_suppressed_patches(axes[row, 2], ex["img"], ex["hm_orig"], ex["sup_ids"])
        show_caption_panel(axes[row, 3], ex["cap_orig"], ex["cap_supp"])
        if row > 0:
            for col in range(4):
                axes[row, col].set_title("")

    plt.suptitle(
        "Suppressing top-20 attended patches (red) shifts the generated caption",
        fontsize=10, y=1.01, fontweight="bold"
    )
    plt.tight_layout(pad=0.6, w_pad=0.4, h_pad=0.6)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "editing_examples.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


def _word_diff(a, b):
    """Number of words in a not in b, as a fraction of len(a)."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa:
        return 0.0
    return len(wa - wb) / len(wa)


def make_transplant_figure(proj, free_dec, guided_dec, tok,
                            test_feats, test_imgs, noun_labels_test=None,
                            n_examples=4, max_scan=500):
    """
    Renders a transplant figure: source image decoded with donor attention plan.

    Scans up to max_scan random (src, donor) pairs, keeping only those where
    the transplanted caption differs from the source caption (at least one word
    change).  Prefers pairs where the source and donor have different noun labels.

    Layout: 4-row × 3-col
      Col 0: source image + its own caption
      Col 1: donor image + its own caption
      Col 2: text panel — source caption vs transplanted caption
    """
    import random
    rng = random.Random(99)
    n_pool = min(80, len(test_feats))

    # Pre-extract oracle plans for all candidate images
    print(f"  Pre-extracting oracle plans for first {n_pool} test images...")
    plans_cache = {}

    def get_plan(i):
        if i not in plans_cache:
            with torch.no_grad():
                X, A0 = extract_oracle_plan(proj, free_dec, test_feats[i])
            cap = decode_with_plan(guided_dec, X, A0, tok)
            plans_cache[i] = (X, A0, cap)
        return plans_cache[i]

    # Generate all pairs; prioritise different-noun pairs
    all_pairs = [(i, j) for i in range(n_pool) for j in range(n_pool) if i != j]
    rng.shuffle(all_pairs)

    good_pairs = []
    checked = 0
    for src_i, don_i in all_pairs:
        if len(good_pairs) >= n_examples or checked >= max_scan:
            break
        checked += 1

        X_src, A0_src, cap_src = get_plan(src_i)
        X_don, A0_don, cap_don = get_plan(don_i)

        # Only filter garbled text on the DISPLAYED src and transplanted captions.
        # Donor's own caption may have <UNK> and still produce a valid transplant.
        if "<UNK>" in cap_src or len(cap_src.split()) < 5:
            continue

        # Skip if src image already used in a selected pair
        if src_i in {p["src_i"] for p in good_pairs}:
            continue

        with torch.no_grad():
            cap_trans = decode_with_plan(guided_dec, X_src, A0_don, tok)

        if "<UNK>" in cap_trans:
            continue

        diff = _word_diff(cap_src, cap_trans)
        if diff == 0.0:
            continue  # caption unchanged — skip

        good_pairs.append({
            "src_i":    src_i,
            "don_i":    don_i,
            "src_img":  test_imgs[src_i],
            "don_img":  test_imgs[don_i],
            "cap_src":  cap_src,
            "cap_don":  cap_don,
            "cap_trans": cap_trans,
            "diff":     diff,
        })
        print(f"    [{len(good_pairs)}] src={src_i} don={don_i}  diff={diff:.2f}")
        print(f"         src:   {cap_src}")
        print(f"         donor: {cap_don}")
        print(f"         trans: {cap_trans}")

    if not good_pairs:
        print(f"  WARNING: no transplant pairs with caption changes found "
              f"in {checked} candidates — saving best-effort figure")
        # Fall back: just show first n_examples pairs regardless
        for src_i, don_i in all_pairs[:n_examples]:
            X_src, A0_src, cap_src = get_plan(src_i)
            X_don, A0_don, cap_don = get_plan(don_i)
            with torch.no_grad():
                cap_trans = decode_with_plan(guided_dec, X_src, A0_don, tok)
            good_pairs.append({
                "src_i": src_i, "don_i": don_i,
                "src_img": test_imgs[src_i], "don_img": test_imgs[don_i],
                "cap_src": cap_src, "cap_don": cap_don, "cap_trans": cap_trans,
                "diff": 0.0,
            })

    # Sort by diff descending — best examples first
    good_pairs.sort(key=lambda x: x["diff"], reverse=True)
    selected = good_pairs[:n_examples]

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.4 * n_rows),
                              gridspec_kw={"width_ratios": [1, 1, 1.6]})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, title in zip(range(3), ["Source image", "Donor image",
                                      "Source → Transplanted"]):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=4)

    for row, ex in enumerate(selected):
        show_image(axes[row, 0], ex["src_img"])
        axes[row, 0].set_xlabel(
            _short(ex["cap_src"], 45), fontsize=7, labelpad=3, color="#333333"
        )
        show_image(axes[row, 1], ex["don_img"])
        axes[row, 1].set_xlabel(
            _short(ex["cap_don"], 45), fontsize=7, labelpad=3, color="#555555"
        )

        ax = axes[row, 2]
        ax.axis("off")
        if row > 0:
            ax.set_title("")
        ax.text(0.04, 0.90, "Source caption:", transform=ax.transAxes,
                fontsize=7.5, fontweight="bold", va="top", color="#222222")
        ax.text(0.04, 0.78, _wrap(ex["cap_src"]), transform=ax.transAxes,
                fontsize=7, va="top", color="#444444", linespacing=1.4)
        ax.text(0.04, 0.50, "Transplanted (donor plan applied to source):",
                transform=ax.transAxes, fontsize=7.5, fontweight="bold",
                va="top", color="#005599")
        ax.text(0.04, 0.36, _wrap(ex["cap_trans"]), transform=ax.transAxes,
                fontsize=7, va="top", color="#005599", linespacing=1.4)

    plt.suptitle(
        "Plan transplant: source features decoded with a different image's attention plan",
        fontsize=10, y=1.01, fontweight="bold"
    )
    plt.tight_layout(pad=0.6, w_pad=0.4, h_pad=0.6)

    out_path = os.path.join(FIG_DIR, "transplant_example.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


def _wrap(s, width=38):
    words = s.split()
    lines, cur = [], []
    for w in words:
        if sum(len(c) + 1 for c in cur) + len(w) > width and cur:
            lines.append(" ".join(cur)); cur = []
        cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _short(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


# ── Single-example figures (clean, paper-ready) ───────────────────────────────

@torch.no_grad()
def make_single_editing_figure(proj, free_dec, denoiser, guided_dec, tok,
                                test_feats, plans_test, test_imgs, noun_labels_test,
                                example_idx=97, k=40):
    """
    Causal intervention figure (Option D):
      Left col  (spans both rows): original image (static)
      Middle col, Row 0: image + original A₀ heatmap overlay
      Middle col, Row 1: image + edited A₀ heatmap overlay (suppressed region cooled)
      Right col,  Row 0: original caption (green)
      Right col,  Row 1: caption after intervention (red)
      Plus a downward arrow in the middle column labelled "intervene".

    Uses saved denoised plans (plans_test).  k=40 patches suppressed.
    """
    feat = test_feats[example_idx]
    img  = test_imgs[example_idx]

    X   = proj(feat.unsqueeze(0).to(DEVICE))
    A0  = plans_test[example_idx].unsqueeze(0).to(DEVICE)

    importance = A0[0].sum(0)                          # [N]
    fg_ids = importance.topk(k).indices

    A0_edit = A0.clone()
    A0_edit[0, :, fg_ids] = 0.0
    A0_edit[0] /= A0_edit[0].sum(-1, keepdim=True).clamp(min=1e-9)

    cap_orig = decode_with_plan(guided_dec, X, A0,      tok)
    cap_edit = decode_with_plan(guided_dec, X, A0_edit, tok)

    hm_orig = plan_to_numpy_heatmap(A0)
    hm_edit = plan_to_numpy_heatmap(A0_edit)

    print(f"  Intervention idx={example_idx}  k={k}")
    print(f"    original: {cap_orig}")
    print(f"    edited:   {cap_edit}")

    # ── Compact side-by-side layout ──────────────────────────────────────
    # Row 0: [orig heatmap] [arrow] [edited heatmap]
    # Row 1: [orig caption (1 line)] [     ] [edited caption + 1-line diff]
    fig = plt.figure(figsize=(6.5, 3.6))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[2.2, 0.32, 2.2],
        height_ratios=[3.0, 1.0],
        hspace=0.06, wspace=0.10,
        left=0.02, right=0.98, top=0.90, bottom=0.02,
    )
    ax_orig_img = fig.add_subplot(gs[0, 0])
    ax_arrow    = fig.add_subplot(gs[0, 1])
    ax_edit_img = fig.add_subplot(gs[0, 2])
    ax_orig_cap = fig.add_subplot(gs[1, 0])
    ax_edit_cap = fig.add_subplot(gs[1, 2])

    show_heatmap(ax_orig_img, img, hm_orig, alpha=0.55,
                 title="Original $A_0$")
    show_heatmap(ax_edit_img, img, hm_edit, alpha=0.55,
                 title="After intervention")

    # Horizontal arrow between images
    ax_arrow.axis("off")
    ax_arrow.annotate(
        "",
        xy=(0.85, 0.55), xytext=(0.15, 0.55),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.8),
    )
    ax_arrow.text(0.50, 0.28, "zero\ntop-40",
                  transform=ax_arrow.transAxes,
                  fontsize=6, ha="center", va="top",
                  color="#555555", style="italic", linespacing=1.3)

    # Compute word-level diff (content words only)
    stopwords = {"a","an","the","of","in","on","at","with","and","is","are",
                 "his","her","their","its","to","up","into","from","by","for",
                 "that","this","has","have","been","was","were","be","some",
                 "there","front","next","top","bottom","side","back","near"}
    def content_words(cap):
        return {w.lower().rstrip(".,!?;:") for w in cap.split()
                if w.lower().rstrip(".,!?;:") not in stopwords and len(w) > 2}

    orig_cw = content_words(cap_orig)
    edit_cw = content_words(cap_edit)
    removed   = sorted(orig_cw - edit_cw)
    added     = sorted(edit_cw - orig_cw)

    def wrap(s, w=34):
        words = s.split()
        lines, cur = [], []
        for word in words:
            if sum(len(c)+1 for c in cur) + len(word) > w and cur:
                lines.append(" ".join(cur)); cur = []
            cur.append(word)
        if cur: lines.append(" ".join(cur))
        return "\n".join(lines)

    # Caption panels — caption text + single compact diff line
    ax_orig_cap.axis("off")
    ax_orig_cap.text(0.03, 0.90, wrap(cap_orig), transform=ax_orig_cap.transAxes,
                     fontsize=7, va="top", color="#116611", linespacing=1.35)

    ax_edit_cap.axis("off")
    ax_edit_cap.text(0.03, 0.90, wrap(cap_edit), transform=ax_edit_cap.transAxes,
                     fontsize=7, va="top", color="#aa1100", linespacing=1.35)

    # Single compact diff line: "✘ doughnut    + banana, jacket, yellow"
    diff_segs = []
    if removed:
        diff_segs.append("✘ " + ", ".join(removed))
    if added:
        diff_segs.append("+ " + ", ".join(added))
    if diff_segs:
        ax_edit_cap.text(0.03, 0.22, "    ".join(diff_segs),
                         transform=ax_edit_cap.transAxes,
                         fontsize=6.5, va="top", color="#333366",
                         bbox=dict(boxstyle="round,pad=0.18",
                                   fc="#eeeeff", ec="#aaaacc", lw=0.5))

    plt.suptitle(
        "Attention plan intervention: $X$ fixed, only $A_0$ changes",
        fontsize=8.5, fontweight="bold", y=0.97,
    )

    out_path = os.path.join(FIG_DIR, "editing_single.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


def _sd_text2img(prompt, seed=42, steps=30, guidance=7.5, size=512):
    """Generate an image from text using cached SD 1.5 weights."""
    from diffusers import StableDiffusionPipeline
    import torch as _torch
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=_torch.float16,
        safety_checker=None,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    gen = _torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance,
                  generator=gen, height=size, width=size)
    img = result.images[0]
    del pipe
    _torch.cuda.empty_cache()
    return img


@torch.no_grad()
def make_single_transplant_figure(proj, free_dec, guided_dec, tok,
                                   test_feats, test_imgs,
                                   src_idx=48, don_idx=11):
    """
    One-row transplant figure showing ONE clear example.
      Col 0: source image (small) + its caption
      Col 1: donor image (small) + its caption
      Col 2: comparison panel — source caption vs transplanted caption
    """
    src_feat = test_feats[src_idx]
    don_feat = test_feats[don_idx]

    X_src, A0_src = extract_oracle_plan(proj, free_dec, src_feat)
    X_don, A0_don = extract_oracle_plan(proj, free_dec, don_feat)

    cap_src   = decode_with_plan(guided_dec, X_src, A0_src, tok)
    cap_don   = decode_with_plan(guided_dec, X_don, A0_don, tok)
    cap_trans = decode_with_plan(guided_dec, X_src, A0_don, tok)

    print(f"  Transplant  src={src_idx}  don={don_idx}")
    print(f"    source:      {cap_src}")
    print(f"    donor:       {cap_don}")
    print(f"    transplanted:{cap_trans}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0),
                             gridspec_kw={"width_ratios": [1, 1, 1.6]})

    SIZE = 160

    def show_small(ax, img_pil, caption, title):
        ax.imshow(np.array(img_pil.resize((SIZE, SIZE), Image.LANCZOS)))
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel(_short(caption, 48), fontsize=7.5, labelpad=4, color="#333333")

    show_small(axes[0], test_imgs[src_idx], cap_src, "Source image")
    show_small(axes[1], test_imgs[don_idx], cap_don, "Donor image")

    ax = axes[2]
    ax.axis("off")
    ax.set_title("Plan transplant result", fontsize=8.5, fontweight="bold", pad=3)

    def wrap(s, w=44):
        words = s.split()
        lines, cur = [], []
        for word in words:
            if sum(len(c)+1 for c in cur) + len(word) > w and cur:
                lines.append(" ".join(cur)); cur = []
            cur.append(word)
        if cur: lines.append(" ".join(cur))
        return "\n".join(lines)

    ax.text(0.04, 0.88, "Source caption:", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", color="#222222")
    ax.text(0.04, 0.76, wrap(cap_src), transform=ax.transAxes,
            fontsize=7.5, va="top", color="#444444", linespacing=1.5)

    ax.text(0.04, 0.46, "After applying donor's attention plan:",
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="top", color="#005599")
    ax.text(0.04, 0.34, wrap(cap_trans), transform=ax.transAxes,
            fontsize=7.5, va="top", color="#005599", linespacing=1.5)

    plt.suptitle(
        "Plan transplant: source image decoded with a different image's attention plan",
        fontsize=9.5, fontweight="bold", y=1.02
    )
    plt.tight_layout(pad=0.5, w_pad=0.6)

    out_path = os.path.join(FIG_DIR, "transplant_single.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


def make_prediction_gap_figure(proj, guided_dec, tok,
                                test_feats, plans_test, test_imgs,
                                example_idx=97, n_inv_steps=200, inv_lr=0.05):
    """
    Two-column figure showing the prediction gap for one test image.
    Left:  denoised plan  → caption  (ROUGE ≈ 0.50)
    Right: inverted plan  → caption  (ROUGE ≈ 0.72)
    """
    from run_plan_inversion import invert_plan, tokenize_caption
    from run_summarization import eval_rouge

    out_path = os.path.join(FIG_DIR, "plan_inversion_gap.pdf")

    feat  = test_feats[example_idx]
    img   = test_imgs[example_idx]
    X     = proj(feat.unsqueeze(0).to(DEVICE))

    # Denoised plan (saved)
    A_den = plans_test[example_idx].unsqueeze(0).to(DEVICE)
    cap_den = decode_with_plan(guided_dec, X, A_den, tok)

    # Reference caption: use the free decoder caption as proxy target
    # (we want a real reference caption for ROUGE, but we only have captions
    #  inside the figure script via the free decoder)
    from run_captioning import CaptionDecoder
    # load reference caption from HuggingFace for this one image
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    test_start = N_IMAGES + VAL_IMAGES
    item = ds[test_start + example_idx]
    refs = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
    ref_cap = refs[0]

    rouge_den = eval_rouge([cap_den], [ref_cap])

    # Invert plan toward reference caption
    tgt_toks = tokenize_caption(tok, ref_cap)
    A_inv = invert_plan(guided_dec, X, tgt_toks, A_init=A_den,
                        n_steps=n_inv_steps, lr=inv_lr, verbose=False)
    cap_inv = decode_with_plan(guided_dec, X, A_inv, tok)
    rouge_inv = eval_rouge([cap_inv], [ref_cap])

    print(f"  Prediction gap  idx={example_idx}")
    print(f"    reference : {ref_cap}")
    print(f"    denoised  : {cap_den}  (ROUGE-1={rouge_den:.3f})")
    print(f"    inverted  : {cap_inv}  (ROUGE-1={rouge_inv:.3f})")

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6.5, 3.8))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[2.5, 0.18, 1.0],
                            hspace=0.08, wspace=0.12)
    ax_img_den  = fig.add_subplot(gs[0, 0])
    ax_img_inv  = fig.add_subplot(gs[0, 1])
    ax_div      = fig.add_subplot(gs[1, :])
    ax_cap_den  = fig.add_subplot(gs[2, 0])
    ax_cap_inv  = fig.add_subplot(gs[2, 1])

    GRID = 16
    img_arr = np.array(pil_resize(img, 224))

    def show_heatmap(ax, A, title, rouge_val, color):
        hm = plan_to_numpy_heatmap(A, GRID)
        ax.imshow(img_arr)
        ax.imshow(hm, cmap=ATTN_CMAP, alpha=0.65,
                  extent=[0, 224, 224, 0], aspect='auto', vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3, color=color)
        ax.text(0.5, 0.04, f"ROUGE-1 = {rouge_val:.3f}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.85))

    show_heatmap(ax_img_den, A_den, "Denoised plan", rouge_den, "#555555")
    show_heatmap(ax_img_inv, A_inv, "Inverted plan",  rouge_inv, "#1a7a42")

    # Arrow separator
    ax_div.axis("off")
    ax_div.annotate("", xy=(0.72, 0.5), xytext=(0.28, 0.5),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#888888",
                                   lw=1.2, mutation_scale=10))
    ax_div.text(0.5, 0.5, "gradient optimisation",
                ha="center", va="center", transform=ax_div.transAxes,
                fontsize=6.5, color="#888888",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    def wrap(s, w=40):
        words = s.split()
        lines, cur = [], []
        for word in words:
            if sum(len(c)+1 for c in cur) + len(word) > w and cur:
                lines.append(" ".join(cur)); cur = []
            cur.append(word)
        if cur: lines.append(" ".join(cur))
        return "\n".join(lines)

    def show_cap(ax, cap, color, label):
        ax.axis("off")
        ax.set_facecolor("#f8f8f8")
        ax.text(0.03, 0.92, label, transform=ax.transAxes,
                fontsize=6.5, fontweight="bold", va="top", color="#444444")
        ax.text(0.03, 0.68, wrap(cap), transform=ax.transAxes,
                fontsize=7.0, va="top", color=color,
                linespacing=1.4)

    show_cap(ax_cap_den, cap_den, "#333333", "Generated:")
    show_cap(ax_cap_inv, cap_inv, "#1a7a42", "Generated:")

    # Overall title
    delta = rouge_inv - rouge_den
    fig.suptitle(f"Prediction gap: denoised plan → oracle inversion  "
                 f"(ROUGE-1 $\\Delta$={delta:+.3f})",
                 fontsize=8.5, y=1.01)

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


def make_qualitative_inversion_figure(proj, guided_dec, tok,
                                       test_feats, plans_test, test_imgs,
                                       example_idxs=(3, 17, 45, 82),
                                       n_inv_steps=200, inv_lr=0.05):
    """
    4-row qualitative panel: image | denoised plan | inverted plan | captions.
    Each row is one test example; target = first MSCOCO reference caption.
    """
    from run_plan_inversion import invert_plan, tokenize_caption
    from run_summarization import eval_rouge
    from datasets import load_dataset

    out_path = os.path.join(FIG_DIR, "qualitative_inversion.pdf")
    test_start = N_IMAGES + VAL_IMAGES

    print("  Loading MSCOCO captions for qualitative examples...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")

    NROWS = len(example_idxs)
    fig = plt.figure(figsize=(8.5, 2.3 * NROWS))
    gs = gridspec.GridSpec(NROWS, 4, figure=fig,
                           width_ratios=[1, 1, 1, 1.9],
                           hspace=0.12, wspace=0.05)

    def wrap(s, w=30):
        words = s.split()
        lines, cur = [], []
        for word in words:
            if sum(len(c) + 1 for c in cur) + len(word) > w and cur:
                lines.append(" ".join(cur)); cur = []
            cur.append(word)
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)

    for row_i, img_idx in enumerate(example_idxs):
        print(f"  Example idx={img_idx} ({row_i + 1}/{NROWS})...")
        feat = test_feats[img_idx]
        img  = test_imgs[img_idx]
        with torch.no_grad():
            X = proj(feat.unsqueeze(0).to(DEVICE))

        A_den   = plans_test[img_idx].unsqueeze(0).to(DEVICE)
        cap_den = decode_with_plan(guided_dec, X, A_den, tok)

        item    = ds[test_start + img_idx]
        refs    = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        ref_cap = refs[0]
        rouge_den = eval_rouge([cap_den], [ref_cap])

        tgt_toks = tokenize_caption(tok, ref_cap)
        A_inv    = invert_plan(guided_dec, X, tgt_toks, A_init=A_den,
                               n_steps=n_inv_steps, lr=inv_lr, verbose=False)
        cap_inv  = decode_with_plan(guided_dec, X, A_inv, tok)
        rouge_inv = eval_rouge([cap_inv], [ref_cap])

        img_arr = np.array(pil_resize(img, 224))
        hm_den  = plan_to_numpy_heatmap(A_den)
        hm_inv  = plan_to_numpy_heatmap(A_inv)

        # Col 0: image
        ax_img = fig.add_subplot(gs[row_i, 0])
        ax_img.imshow(img_arr)
        ax_img.axis("off")
        if row_i == 0:
            ax_img.set_title("Image", fontsize=7.5, fontweight="bold", pad=3)

        # Col 1: denoised plan heatmap
        ax_den = fig.add_subplot(gs[row_i, 1])
        ax_den.imshow(img_arr)
        ax_den.imshow(hm_den, cmap=ATTN_CMAP, alpha=0.65,
                      extent=[0, 224, 224, 0], aspect="auto", vmin=0, vmax=1)
        ax_den.axis("off")
        if row_i == 0:
            ax_den.set_title("Denoised plan", fontsize=7.5, fontweight="bold", pad=3)
        ax_den.text(0.5, 0.04, f"R-1={rouge_den:.3f}",
                    transform=ax_den.transAxes, ha="center", va="bottom",
                    fontsize=6.5, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#555555", ec="none", alpha=0.85))

        # Col 2: inverted plan heatmap
        ax_inv = fig.add_subplot(gs[row_i, 2])
        ax_inv.imshow(img_arr)
        ax_inv.imshow(hm_inv, cmap=ATTN_CMAP, alpha=0.65,
                      extent=[0, 224, 224, 0], aspect="auto", vmin=0, vmax=1)
        ax_inv.axis("off")
        if row_i == 0:
            ax_inv.set_title("Inverted plan", fontsize=7.5, fontweight="bold", pad=3)
        ax_inv.text(0.5, 0.04, f"R-1={rouge_inv:.3f}",
                    transform=ax_inv.transAxes, ha="center", va="bottom",
                    fontsize=6.5, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#1a7a42", ec="none", alpha=0.85))

        # Col 3: text comparison panel
        ax_txt = fig.add_subplot(gs[row_i, 3])
        ax_txt.axis("off")
        ax_txt.set_facecolor("#f5f5f5")
        if row_i == 0:
            ax_txt.set_title("Captions", fontsize=7.5, fontweight="bold", pad=3)
        ax_txt.text(0.04, 0.96, "Target:", transform=ax_txt.transAxes,
                    fontsize=5.8, fontweight="bold", va="top", color="#333333")
        ax_txt.text(0.04, 0.84, wrap(ref_cap), transform=ax_txt.transAxes,
                    fontsize=6.0, va="top", color="#222222", linespacing=1.3)
        ax_txt.text(0.04, 0.54, "Denoised:", transform=ax_txt.transAxes,
                    fontsize=5.8, fontweight="bold", va="top", color="#555555")
        ax_txt.text(0.04, 0.42, wrap(cap_den), transform=ax_txt.transAxes,
                    fontsize=6.0, va="top", color="#555555", linespacing=1.3)
        ax_txt.text(0.04, 0.14, "Inverted:", transform=ax_txt.transAxes,
                    fontsize=5.8, fontweight="bold", va="top", color="#1a7a42")
        ax_txt.text(0.04, 0.02, wrap(cap_inv), transform=ax_txt.transAxes,
                    fontsize=6.0, va="top", color="#1a7a42", linespacing=1.3)

    fig.suptitle(
        "Qualitative plan inversion: denoised vs. oracle-inverted plans",
        fontsize=9, fontweight="bold", y=1.005,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tok = load_tokenizer()
    print(f"  Vocab size: {len(tok.vocab)}")

    # ── Models ─────────────────────────────────────────────────────────────
    print("Loading model checkpoints...")
    proj, free_dec, denoiser, guided_dec = load_models(tok)

    # ── Patch features ─────────────────────────────────────────────────────
    print("Loading cached patch features...")
    patch_feats = torch.load(CACHE_PATH, weights_only=True)  # [15000, N, D_enc]
    test_start  = N_IMAGES + VAL_IMAGES
    test_feats  = patch_feats[test_start: test_start + TEST_IMAGES]
    print(f"  test_feats: {test_feats.shape}")

    # ── Saved plans ────────────────────────────────────────────────────────
    print("Loading saved plans...")
    plans_test = torch.load(
        os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True
    )  # [1000, L_CAP, N_PATCHES]
    print(f"  plans_test: {plans_test.shape}")

    # ── Noun labels ────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, "noun_labels_test.json")) as f:
        noun_labels_test = json.load(f)

    # ── Test images from HuggingFace ───────────────────────────────────────
    print("Loading MSCOCO test images from HuggingFace (first-time: slow)...")
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    test_imgs = []
    for item in ds.select(range(test_start, test_start + TEST_IMAGES)):
        img = item["jpg"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        test_imgs.append(img)
    print(f"  Loaded {len(test_imgs)} test images")

    # ── Figure 1: single editing example (foreground → background shift) ─
    print("\nGenerating editing_single.pdf ...")
    make_single_editing_figure(
        proj, free_dec, denoiser, guided_dec, tok,
        test_feats, plans_test, test_imgs, noun_labels_test,
        example_idx=97, k=40,
    )

    # ── Figure 2: single transplant example ────────────────────────────────
    print("\nGenerating transplant_single.pdf ...")
    make_single_transplant_figure(
        proj, free_dec, guided_dec, tok,
        test_feats, test_imgs,
        src_idx=48, don_idx=11,
    )

    # ── Figure 3: prediction gap (plan inversion) ──────────────────────────
    print("\nGenerating plan_inversion_gap.pdf ...")
    make_prediction_gap_figure(
        proj, guided_dec, tok,
        test_feats, plans_test, test_imgs,
        example_idx=97,
    )

    # ── Figure 4: qualitative inversion panel ──────────────────────────────
    print("\nGenerating qualitative_inversion.pdf ...")
    make_qualitative_inversion_figure(
        proj, guided_dec, tok,
        test_feats, plans_test, test_imgs,
        example_idxs=(3, 17, 45, 82),
    )

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
