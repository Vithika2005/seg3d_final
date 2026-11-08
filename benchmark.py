"""Latency / memory benchmark: CPU vs Apple MPS (Metal) for the full pipeline + splat rendering."""
import argparse, csv, glob, gc, time
import numpy as np, psutil, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from geom3d import camera_path, fill_holes, render_splats
from seg3d import Seg3DPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="glob, e.g. 'samples/*.jpg'")
ap.add_argument("--vocab", default="chair, table, sofa, lamp, person")
a = ap.parse_args()
files = sorted(glob.glob(a.images))[:10]
assert files, "no images found"
rows = []
for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
    pipe = Seg3DPipeline(dev)
    pipe.build(Image.open(files[0]), a.vocab)                      # warm-up
    T, S = [], []
    for f in files:
        sc = pipe.build(Image.open(f), a.vocab); T.append(sc["timings"])
        H, W = sc["rgb"].shape[:2]; R, c = camera_path("orbit", [0, 0, 3], 12)[3]
        t0 = time.perf_counter()
        img, cov = render_splats(sc["xyz"].reshape(-1, 3), sc["rgb"].reshape(-1, 3), sc["K"], R, c, (W, H))
        fill_holes(img, cov); S.append((time.perf_counter() - t0) * 1000)
    mem = torch.mps.current_allocated_memory() / 2**20 if dev == "mps" else psutil.Process().memory_info().rss / 2**20
    keys = sorted(set().union(*[t.keys() for t in T]))
    rows.append({"device": dev, **{k: float(np.mean([t.get(k, 0) for t in T])) for k in keys},
                 "splat_render_ms(numpy,CPU)": float(np.mean(S)), "mem_MB": mem})
    del pipe; gc.collect()
    if dev == "mps": torch.mps.empty_cache()
fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "device", k))
with open("results/benchmark.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
stages = [k for k in rows[0] if k in ("detect", "segment", "segment_free", "depth", "lift", "clip")]
x = np.arange(len(stages))
for i, r in enumerate(rows):
    plt.bar(x + i * 0.35, [r.get(s, 0) for s in stages], 0.35, label=r["device"])
plt.xticks(x + 0.175, stages, rotation=20); plt.ylabel("ms / image"); plt.legend(); plt.title("Per-stage latency")
plt.savefig("results/benchmark.png", dpi=200, bbox_inches="tight"); print(rows)
