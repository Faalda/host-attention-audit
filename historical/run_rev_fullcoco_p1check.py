"""run_rev_fullcoco_p1check.py — E9 side check: free-captioner quality on the 2,000 held-out val images
for Phase-1 snapshots (epochs 10..35) vs the final epoch-40 model (token CE overfits after ep 7)."""
from run_rev_fullcoco_eval import *
torch.backends.cuda.matmul.allow_tf32 = True
tok = tok_from_vocab(json.load(open(os.path.join(FC, "tok_vocab.json"))))
caps = load_caps("train"); n_tr = len(caps) - N_VAL_HOLD
feats = GPUFeats("train", n=N_VAL_HOLD, start=n_tr); refs = caps[n_tr:]
out = {}
for ep in [10, 15, 20, 30, 40]:
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES).to(DEVICE)
    if ep == 40:
        proj.load_state_dict(torch.load(os.path.join(FC, "model_proj.pt"), weights_only=True)); free.load_state_dict(torch.load(os.path.join(FC, "model_free.pt"), weights_only=True))
    else:
        st = torch.load(os.path.join(CKPT, "phase1_snap", f"ep{ep}.pt"), weights_only=False)["models"]
        proj.load_state_dict(st["proj"]); free.load_state_dict(st["dec"])
    freeze(proj, free)
    X = project(proj, feats)
    c = [tok.dec(r) for r in gen_tokens(free, X)]
    out[ep] = summ(score(c, refs)); print(ep, out[ep], flush=True)
json.dump(out, open(os.path.join(FC, "phase1_snapshot_val_check.json"), "w"), indent=1)
