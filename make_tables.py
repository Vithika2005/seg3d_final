"""Collect results/*.json into markdown tables (results/tables.md) and a precision/recall plot (results/pr_curve.png).
Usage: python make_tables.py"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
load = lambda n: json.load(open(f"{R}/{n}")) if os.path.exists(f"{R}/{n}") else None
f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
L = []

# ---- Experiment 1: NYU, box-threshold sweep
L += ["## Experiment 1 - NYU Depth V2 (Depth Anything V2 Small, relative depth aligned to GT scale/shift; detector-only)", "",
      "| box_th | AbsRel (lower better) | delta1 | mask mIoU | recall@0.5 | precision@0.5 | 3D box IoU (matched) | GT instances |",
      "|---|---|---|---|---|---|---|---|"]
pts = []
for th, fn in [("0.25", "nyu_small_025.json"), ("0.35", "nyu_small_035.json"), ("0.45", "nyu_small_045.json")]:
    d = load(fn)
    if d is None:
        L.append(f"| {th} | missing {fn} | | | | | | |"); continue
    L.append(f"| {th} | {f(d['depth_AbsRel'])} | {f(d['depth_delta1'])} | {f(d['mask_mIoU'])} | {f(d['recall@0.5'])} | "
             f"{f(d['precision@0.5'])} | {f(d['mean_3D_box_IoU_of_matched'])} | {d['num_gt_instances']} |")
    pts.append((th, d["recall@0.5"], d["precision@0.5"]))
if pts:
    plt.figure(figsize=(4.5, 4)); plt.plot([p[1] for p in pts], [p[2] for p in pts], "o-")
    for th, r, p in pts:
        plt.annotate(f"box_th={th}", (r, p), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Recall@0.5"); plt.ylabel("Precision@0.5"); plt.title("Precision / recall trade-off"); plt.grid(alpha=.3)
    plt.savefig(f"{R}/pr_curve.png", dpi=200, bbox_inches="tight")
L += ["", "Note: precision undercounts because NYU labels are not exhaustive; 3D IoU is axis-aligned and uses GT-aligned depth.", ""]

# ---- Experiment 3: depth backbone
L += ["## Depth backbone (box_th 0.35)", "",
      "| Backbone | AbsRel | delta1 | 3D box IoU | depth stage (ms/img) | total (ms/img) |", "|---|---|---|---|---|---|"]
for name, fn in [("Depth Anything V2 Small", "nyu_small_035.json"), ("Depth Anything V2 Base", "nyu_base_035.json")]:
    d = load(fn)
    if d is None:
        L.append(f"| {name} | missing {fn} | | | | |"); continue
    ms = d.get("mean_ms", {})
    L.append(f"| {name} | {f(d['depth_AbsRel'])} | {f(d['depth_delta1'])} | {f(d['mean_3D_box_IoU_of_matched'])} | "
             f"{f(ms.get('depth'), 0)} | {f(ms.get('total'), 0)} |")
L += ["", f"Latency measured during evaluation on device `{(load('nyu_small_035.json') or {}).get('device', '?')}`, first image excluded (warm-up).", ""]

# ---- Experiment 2: retrieval
fmt = lambda d: f"{d['top1']:.2f} ({d['top1_ci95'][0]:.2f}-{d['top1_ci95'][1]:.2f})"
c = load("queries_combined.json")
L += ["## Experiment 2a - Retrieval / spatial reasoning per query type (combined scorer)", "",
      "| Query type | n | top-1 (95% CI) | any-hit | avg objects returned |", "|---|---|---|---|---|"]
if c:
    for k, d in {**c["per_type"], "ALL": c["overall"]}.items():
        L.append(f"| {k} | {d['n']} | {fmt(d)} | {d['any_hit']:.2f} | {d['avg_returned']:.1f} |")
else:
    L.append("| missing queries_combined.json | | | | |")
L += ["", "## Experiment 2b - Ablation", "", "| Variant | n | top-1 (95% CI) | any-hit | avg returned | out-of-vocab top-1 |", "|---|---|---|---|---|---|"]
for name, fn in [("label match only", "queries_label.json"), ("CLIP relevancy only", "queries_clip.json"),
                 ("combined (default)", "queries_combined.json"), ("combined, no free segments", "queries_combined_nofree.json")]:
    d = load(fn)
    if d is None:
        L.append(f"| {name} | missing {fn} | | | | |"); continue
    o = d["overall"]; oov = d.get("per_tag", {}).get("oov")
    L.append(f"| {name} | {o['n']} | {fmt(o)} | {o['any_hit']:.2f} | {o['avg_returned']:.1f} | {fmt(oov) + f' (n={oov['n']})' if oov else '-'} |")
# ---- Roboflow custom dataset
L += ["## Custom Roboflow dataset (COCO-style AP, same metric code for both methods)", "",
      "| Method | box AP50 | box AP | mask AP50 | mask AP | mask P@0.5 | mask R@0.5 | ms / image |", "|---|---|---|---|---|---|---|---|"]
for name, fn in [("Ours (zero-shot)", "roboflow_ours.json"), ("YOLO-seg (trained on our train split)", "roboflow_yolo.json")]:
    d = load(fn)
    if d is None:
        L.append(f"| {name} | missing {fn} | | | | | | |"); continue
    L.append(f"| {name} | {f(d['box']['AP50'])} | {f(d['box']['AP'])} | {f(d['mask']['AP50'])} | {f(d['mask']['AP'])} | "
             f"{f(d['mask']['precision50'])} | {f(d['mask']['recall50'])} | {f(d.get('mean_ms_per_image'), 0)} |")
L.append("")

open(f"{R}/tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\nsaved {R}/tables.md and {R}/pr_curve.png")
