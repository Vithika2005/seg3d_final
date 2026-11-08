"""Quantitative evaluation on NYU Depth V2: depth, 2D instance masks, 3D boxes (detector mode)."""
import argparse, json
import numpy as np
from PIL import Image
from tqdm import tqdm
from geom3d import backproject, box3d_iou, trimmed_box
from nyu import NYULabeled, NYU_VOCAB
from seg3d import Seg3DPipeline


def mask_iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stride", type=int, default=29); ap.add_argument("--device", default=None)
    ap.add_argument("--depth-mode", default="relative", choices=["relative", "relative_base", "indoor"])
    ap.add_argument("--box-th", type=float, default=0.35)
    ap.add_argument("--out", default="results/metrics.json")
    a = ap.parse_args()
    ds, pipe = NYULabeled(a.mat), Seg3DPipeline(a.device, a.depth_mode)
    prompt = ", ".join(NYU_VOCAB)
    T = []; skipped = 0
    M = {k: [] for k in ["absrel", "delta1", "gt_iou", "gt_hit", "pred_hit", "iou3d"]}
    for k in tqdm(range(a.n)):
        try:
            s = ds.get((k * a.stride) % ds.n)
        except Exception as ex:
            skipped += 1; print(f"skipping unreadable image {(k * a.stride) % ds.n}: {ex}"); continue
        res = pipe.run(Image.fromarray(s["rgb"]), prompt, box_th=a.box_th, gt_depth=s["depth"], K=s["K"])
        T.append(res["timings"])
        p, g = res["depth"], s["depth"]; v = g > 1e-3
        M["absrel"].append(np.mean(np.abs(p[v] - g[v]) / g[v]))
        M["delta1"].append(np.mean(np.maximum(p[v] / g[v], g[v] / p[v]) < 1.25))
        xyz_gt = backproject(g, s["K"])
        key = np.where((s["instances"] > 0) & (s["labels"] > 0), s["labels"] * 1000 + s["instances"], 0)
        gts = []
        for kv in np.unique(key):
            m = key == kv
            if kv and ds.names[kv // 1000 - 1] in NYU_VOCAB and m.sum() >= 1500:
                gts.append((ds.names[kv // 1000 - 1], m))
        insts = res["instances"]
        for cls, m in gts:
            cands = [(mask_iou(i.mask, m), i) for i in insts if i.label == cls]
            best, bi = max(cands, key=lambda x: x[0]) if cands else (0.0, None)
            M["gt_iou"].append(best); M["gt_hit"].append(best >= 0.5)
            if bi is not None and best >= 0.5 and (m & v).sum() > 50:
                _, gb = trimmed_box(xyz_gt[m & v]); M["iou3d"].append(box3d_iou(bi.box3d, gb))
        for i in insts:
            ious = [mask_iou(i.mask, m) for c, m in gts if c == i.label]
            M["pred_hit"].append(bool(ious) and max(ious) >= 0.5)
    tt = T[1:] or T          # first image = warm-up, excluded from timing
    mean_ms = {k: float(np.mean([t[k] for t in tt])) for k in tt[0]}
    out = {"mean_ms": mean_ms, "images": a.n - skipped, "skipped_images": skipped, "device": str(pipe.device), "depth_mode": a.depth_mode, "box_th": a.box_th,
           "depth_AbsRel": float(np.mean(M["absrel"])), "depth_delta1": float(np.mean(M["delta1"])),
           "mask_mIoU": float(np.mean(M["gt_iou"])), "recall@0.5": float(np.mean(M["gt_hit"])),
           "precision@0.5": float(np.mean(M["pred_hit"])) if M["pred_hit"] else 0.0,
           "mean_3D_box_IoU_of_matched": float(np.mean(M["iou3d"])) if M["iou3d"] else 0.0,
           "num_gt_instances": len(M["gt_iou"])}
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
