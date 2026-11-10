"""
run_rev_fullcoco_features.py — revision experiment E9 (R2: stronger captioner on full-scale COCO).

Extracts DINOv2 ViT-B/14 patch features (256 x 768, CLS dropped) for
  train : clip-benchmark/wds_mscoco_captions split="train"  (82,783 images = COCO train2014)
  test  : split="test"                                         (5,000 images = Karpathy test)
with exactly the preprocessing of run_cap_improved.extract_patch_features (short side -> 256
bicubic, centre-crop 224, ImageNet mean/std, fp32 model, last_hidden_state[:, 1:]).

Storage (disk budget): int8 with a per-(image, channel) absmax scale over the 256 patches,
  x ~= q.astype(f32) * scale,  q: int8 [N,256,768] (.npy memmap),  scale: fp16 [N,768].
Measured relative L2 error on the old 15k cache: 0.62%.
Also writes the first 5 captions per image (JSON). Resumable: progress is tracked per chunk.

Usage: CUDA_VISIBLE_DEVICES=0 python run_rev_fullcoco_features.py
"""
import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PILImage

OUT = "/home/faldabbas/myproject/attn_diffusion_gen/results/full_coco"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_CAP_PER = 5
BS = 128


def preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), PILImage.BICUBIC)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1)


class ImgDS(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return preprocess(self.ds[i]["jpg"])


def run_split(split):
    from datasets import load_dataset
    from transformers import AutoModel
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split=split)
    n = len(ds)
    cap_path = os.path.join(OUT, f"caps_{split}.json")
    if not os.path.exists(cap_path):
        caps = [[c.strip() for c in t.strip().split("\n") if c.strip()][:N_CAP_PER] for t in list(ds["txt"])]
        assert all(len(c) == N_CAP_PER for c in caps)
        json.dump({"keys": list(ds["__key__"]), "caps": caps}, open(cap_path, "w"))
    qpath = os.path.join(OUT, f"dino_{split}_q.npy")
    spath = os.path.join(OUT, f"dino_{split}_scale.npy")
    prog = os.path.join(OUT, f"dino_{split}_progress.json")
    mode = "r+" if os.path.exists(qpath) else "w+"
    Q = np.lib.format.open_memmap(qpath, mode=mode, dtype=np.int8, shape=(n, 256, 768))
    S = np.lib.format.open_memmap(spath, mode=mode, dtype=np.float16, shape=(n, 768))
    done = json.load(open(prog))["done"] if os.path.exists(prog) else 0
    print(f"[{split}] n={n}  resuming from {done}", flush=True)
    if done >= n:
        return
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    loader = DataLoader(ImgDS(ds.select(range(done, n))), batch_size=BS, num_workers=20, shuffle=False)
    i = done; t0 = time.time()
    with torch.no_grad():
        for px in loader:
            f = model(pixel_values=px.to(DEVICE)).last_hidden_state[:, 1:, :].float()   # [B,256,768]
            sc = (f.abs().amax(1) / 127.0).clamp(min=1e-4).half()                     # [B,768]
            q = (f / sc.float().unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
            b = f.shape[0]
            Q[i:i + b] = q.cpu().numpy(); S[i:i + b] = sc.cpu().numpy()
            i += b
            if (i // BS) % 50 == 0 or i == n:
                Q.flush(); S.flush()
                json.dump({"done": i}, open(prog, "w"))
                print(f"[{split}] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    Q.flush(); S.flush(); json.dump({"done": n}, open(prog, "w"))


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["test", "train"]):
        run_split(sp)
    print("done")
