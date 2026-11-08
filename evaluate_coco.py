"""Evaluate detection + instance segmentation on a Roboflow *COCO segmentation* export.
Export layout: <root>/<split>/_annotations.coco.json + images (Roboflow splits: train / valid / test).
Ours (zero-shot, no training):  python evaluate_coco.py --root roboflow_coco --split test --out results/roboflow_ours.json
YOLO baseline (trained by you): python evaluate_coco.py --root roboflow_coco --split test --yolo runs/segment/train/weights/best.pt --out results/roboflow_yolo.json
Both use the same metric code (cocoeval_lite.py), so the numbers are comparable."""
import argparse, json, os, collections
import numpy as np
from PIL import Image
from tqdm import tqdm
from cocoeval_lite import evaluate_predictions, polygons_to_mask

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--split", default="test")
ap.add_argument("--box-th", type=float, default=0.25); ap.add_argument("--text-th", type=float, default=0.25)
ap.add_argument("--depth-mode", default="relative"); ap.add_argument("--device", default=None)
ap.add_argument("--yolo", default=None, help="path to trained Ultralytics weights (best.pt)")
ap.add_argument("--out", default="results/roboflow_ours.json")
a = ap.parse_args()

folder = os.path.join(a.root, a.split)
coco = json.load(open(os.path.join(folder, "_annotations.coco.json")))
cat_name = {c["id"]: c["name"].strip().lower() for c in coco["categories"]}
used = sorted({x["category_id"] for x in coco["annotations"]})        # Roboflow adds a dummy category 0: ignore unused ones
names = [cat_name[i] for i in used]
images = {im["id"]: im for im in coco["images"]}
by_img = collections.defaultdict(list)
skipped = 0
for x in coco["annotations"]:
    if x.get("iscrowd", 0) or not isinstance(x.get("segmentation"), list) or not x["segmentation"]:
        skipped += 1; continue
    by_img[x["image_id"]].append(x)
print(f"{len(images)} images, classes: {names}" + (f", skipped {skipped} annotations without polygons" if skipped else ""))

gts = {n: collections.defaultdict(list) for n in names}
for iid, anns in by_img.items():
    im = images[iid]
    for x in anns:
        bx, by, bw, bh = x["bbox"]
        gts[cat_name[x["category_id"]]][iid].append({"box": (bx, by, bx + bw, by + bh),
                                                      "mask": polygons_to_mask(x["segmentation"], im["height"], im["width"])})


def map_label(label):
    l = label.strip().lower()
    if l in names:
        return l
    toks = l.replace(".", " ").split()
    for n in names:
        if n in toks or n in l:
            return n
    return None


if a.yolo:
    from ultralytics import YOLO
    import torch
    model = YOLO(a.yolo); dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
else:
    from seg3d import Seg3DPipeline
    pipe = Seg3DPipeline(a.device, a.depth_mode); vocab = ", ".join(names)

preds, times = [], []
for iid, im in tqdm(images.items()):
    path = os.path.join(folder, im["file_name"]); img = Image.open(path).convert("RGB"); W, H = img.size
    if a.yolo:
        import time
        t0 = time.perf_counter(); r = model.predict(path, conf=0.001, device=dev, verbose=False)[0]; times.append((time.perf_counter() - t0) * 1000)
        if r.masks is None:
            continue
        for k in range(len(r.boxes)):
            cls = r.names[int(r.boxes.cls[k])].strip().lower()
            if cls in names:
                preds.append({"img": iid, "cls": cls, "score": float(r.boxes.conf[k]), "box": tuple(r.boxes.xyxy[k].tolist()),
                              "mask": polygons_to_mask([r.masks.xy[k].flatten().tolist()], H, W)})
    else:
        res = pipe.run(img, vocab, box_th=a.box_th, text_th=a.text_th); times.append(res["timings"]["total"])
        for ins in res["instances"]:
            cls = map_label(ins.label)
            if cls:
                preds.append({"img": iid, "cls": cls, "score": ins.score, "box": tuple(ins.box2d.tolist()), "mask": ins.mask})

m = evaluate_predictions(preds, gts)
tt = times[1:] or times
out = {"method": "yolo" if a.yolo else "ours_zero_shot", "split": a.split, "images": len(images), "classes": names,
       "n_gt": int(sum(len(v) for g in gts.values() for v in g.values())), "n_pred": len(preds),
       "box": m["box"], "mask": m["mask"], "per_class": m["per_class"], "mean_ms_per_image": float(np.mean(tt)) if tt else None}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
print(json.dumps({k: out[k] for k in ("method", "images", "n_gt", "n_pred", "box", "mask", "mean_ms_per_image")}, indent=2))
