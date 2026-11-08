"""COCO-style detection / instance-segmentation metrics without pycocotools (numpy + PIL only).
AP uses 101-point interpolation, IoU thresholds 0.50:0.05:0.95, greedy matching in score order."""
import numpy as np
from PIL import Image, ImageDraw

IOUS = np.arange(0.5, 0.96, 0.05)


def polygons_to_mask(polys, h, w):
    m = Image.new("L", (w, h), 0); d = ImageDraw.Draw(m)
    for p in polys:
        if len(p) >= 6:
            d.polygon(list(zip(p[0::2], p[1::2])), fill=1)
    return np.array(m, bool)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def class_ap(preds, gts, iou_fn):
    """preds: list of (img_id, score, obj); gts: img_id -> list of obj. Returns one (AP, tp, n_pred, n_gt) per IoU threshold."""
    npos = sum(len(v) for v in gts.values())
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    ious = [[iou_fn(preds[i][2], g) for g in gts.get(preds[i][0], [])] for i in range(len(preds))]
    out = []
    for thr in IOUS:
        taken = {k: [False] * len(v) for k, v in gts.items()}
        tp = np.zeros(len(order))
        for rank, i in enumerate(order):
            img = preds[i][0]; best, bj = -1.0, -1
            for j, iou in enumerate(ious[i]):
                if not taken[img][j] and iou >= thr and iou > best:
                    best, bj = iou, j
            if bj >= 0:
                taken[img][bj] = True; tp[rank] = 1
        ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
        rec = ctp / npos if npos else ctp
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        for k in range(len(prec) - 2, -1, -1):
            prec[k] = max(prec[k], prec[k + 1])
        if len(prec):
            idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
            ap = float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))
        else:
            ap = 0.0
        out.append((ap, int(tp.sum()), len(order), npos))
    return out


def evaluate_predictions(preds, gts):
    """preds: list of dict(img, cls, score, box, mask); gts: cls -> img -> list of dict(box, mask).
    Returns {'box': {...}, 'mask': {...}, 'per_class': {...}}; metrics averaged over classes that have ground truth."""
    res = {"per_class": {}}
    for kind, fn in (("box", box_iou), ("mask", mask_iou)):
        ap50, ap75, ap, tps, npred, ngt = [], [], [], 0, 0, 0
        for cls, gimg in gts.items():
            P = [(p["img"], p["score"], p[kind]) for p in preds if p["cls"] == cls]
            G = {img: [g[kind] for g in lst] for img, lst in gimg.items()}
            r = class_ap(P, G, fn)
            ap50.append(r[0][0]); ap75.append(r[5][0]); ap.append(float(np.mean([x[0] for x in r])))
            tps += r[0][1]; npred += r[0][2]; ngt += r[0][3]
            res["per_class"].setdefault(cls, {}).update({f"{kind}_AP50": r[0][0], f"{kind}_AP": ap[-1]})
        res[kind] = {"AP50": float(np.mean(ap50)) if ap50 else 0.0, "AP75": float(np.mean(ap75)) if ap75 else 0.0,
                     "AP": float(np.mean(ap)) if ap else 0.0,
                     "precision50": tps / npred if npred else 0.0, "recall50": tps / ngt if ngt else 0.0}
    return res
