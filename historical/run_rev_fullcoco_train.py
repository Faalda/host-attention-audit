"""
run_rev_fullcoco_train.py — revision experiment E9: 3-phase recipe of run_save_plans.py at full COCO scale.

  train images : split="train" [0, 80783)      (COCO train2014 minus the last 2,000)
  val images   : split="train" [80783, 82783)  (held out from Phase 1; used for val loss only there.
                 As in run_save_plans.py, Phases 2/3 train on train+val plans.)
  vocab        : 8,000 words from the 82,783 training-split images' captions (5 per image)

Phase 1 free captioner  -> Phase 2 CFG denoiser on teacher-forced plans (argmax targets, masked rows)
-> Phase 3 guided decoder (train_guided_mixed: 50% softmax(10*A0), 50% maskgit K in {1,2,4,8}).
Checkpoints every epoch to results/full_coco/ckpt/ (resumable); final weights in results/full_coco/.

Usage:
  CUDA_VISIBLE_DEVICES=0 nohup python run_rev_fullcoco_train.py --ep1 E --ep2 E --ep3 E --bs B --lr LR > logs/... &
  python run_rev_fullcoco_train.py --bench      # time a few batches of each phase
"""
import argparse
from run_rev_fullcoco_common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep1", type=int, default=8)
    ap.add_argument("--ep2", type=int, default=8)
    ap.add_argument("--ep3", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    set_seed(args.seed)

    caps = load_caps("train")
    n_all = len(caps); n_tr = n_all - N_VAL_HOLD
    vpath = os.path.join(FC, "tok_vocab.json")
    if os.path.exists(vpath):
        tok = tok_from_vocab(json.load(open(vpath)))
    else:
        tok = build_tok(caps, 8000)
        json.dump(tok.vocab, open(vpath, "w"))
    print(f"vocab {len(tok.vocab)}  train imgs {n_tr}  val imgs {N_VAL_HOLD}", flush=True)
    cfg = vars(args) | {"n_train_images": n_tr, "n_val_images": N_VAL_HOLD, "vocab": len(tok.vocab),
                        "orig_recipe": {"batch": 32, "lr": 3e-4, "epochs": [40, 60, 50], "n_images_tv": 14000}}
    json.dump(cfg, open(os.path.join(FC, "train_config.json"), "w"), indent=1)

    t0 = time.time()
    feats = GPUFeats("train")
    print(f"features on GPU: {tuple(feats.q.shape)} int8 ({time.time() - t0:.0f}s)", flush=True)
    tr_img, tr_ids = encode_pairs(caps, tok, range(n_tr))
    va_img, va_ids = encode_pairs(caps, tok, range(n_tr, n_all))
    tv_img, tv_ids = encode_pairs(caps, tok, range(n_all))
    print(f"pairs: train {len(tr_img)}  val {len(va_img)}  train+val {len(tv_img)}", flush=True)

    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES).to(DEVICE)
    den = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    gdec = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    if args.bench:
        nb = 40
        sub = torch.arange(nb * args.bs, device=DEVICE)
        for ph in (1, 2, 3):
            torch.cuda.synchronize(); t = time.time()
            if ph == 1:
                train_free_cap_stream(proj, free, feats, tr_img[sub], tr_ids[sub], 1, args.bs, args.lr, name="bench1")
            elif ph == 2:
                train_cfg_denoiser_stream(den, proj, free, feats, tv_img[sub], tv_ids[sub], 1, args.bs, args.lr, name="bench2")
            else:
                train_guided_mixed_stream(gdec, den, proj, free, feats, tv_img[sub], tv_ids[sub], 1, args.bs, args.lr, name="bench3")
            torch.cuda.synchronize(); dt = time.time() - t
            print(f"phase {ph}: {dt / nb * 1000:.1f} ms/batch -> {dt / nb * len(tv_img) / args.bs / 60:.1f} min/epoch", flush=True)
        for f in ("bench1", "bench2", "bench3"):
            os.remove(os.path.join(CKPT, f"{f}.pt"))
        print(f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        return

    print("Phase 1: free captioner", flush=True)
    train_free_cap_stream(proj, free, feats, tr_img, tr_ids, args.ep1, args.bs, args.lr, val=(va_img, va_ids))
    torch.save(proj.state_dict(), os.path.join(FC, "model_proj.pt"))
    torch.save(free.state_dict(), os.path.join(FC, "model_free.pt"))
    freeze(proj, free)

    print("Phase 2: CFG denoiser", flush=True)
    train_cfg_denoiser_stream(den, proj, free, feats, tv_img, tv_ids, args.ep2, args.bs, args.lr)
    torch.save(den.state_dict(), os.path.join(FC, "model_denoiser.pt"))
    freeze(den)

    print("Phase 3: guided decoder", flush=True)
    train_guided_mixed_stream(gdec, den, proj, free, feats, tv_img, tv_ids, args.ep3, args.bs, args.lr)
    torch.save(gdec.state_dict(), os.path.join(FC, "model_guided.pt"))
    print(f"gates: {gdec.gate_values()}  total {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
