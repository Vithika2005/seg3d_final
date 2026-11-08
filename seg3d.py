"""Core pipeline: text-prompted detection + SAM instance masks (+ SAM grid 'free' segments)
-> monocular depth -> 3D lifting -> CLIP features per instance.
Grounding DINO + SAM + Depth Anything V2 + CLIP (all via Hugging Face transformers; PyTorch MPS)."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # unsupported MPS ops fall back to CPU
import time

import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor)

from geom3d import (Instance, PALETTE, aabb_gap, backproject, box3d_iou, default_K, disparity_to_depth,  # noqa: F401
                    draw_overlay, mask_box, mask_iou_matrix, nms_masks, obb_9dof, save_scene_ply,
                    scene_to_json, trimmed_box)

DET_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"
DEPTH_IDS = {"relative": "depth-anything/Depth-Anything-V2-Small-hf",
             "relative_base": "depth-anything/Depth-Anything-V2-Base-hf",
             "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
             "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"}
DEFAULT_VOCAB = ["chair", "table", "sofa", "bed", "desk", "cabinet", "shelf", "bookshelf", "door", "window", "wall",
                 "floor", "ceiling", "lamp", "pillow", "cushion", "curtain", "picture frame", "painting", "mirror",
                 "tv", "monitor", "laptop", "computer", "keyboard", "plant", "vase", "book", "bottle", "cup", "bowl",
                 "box", "bag", "clock", "rug", "refrigerator", "sink", "toilet", "stove", "person", "tree", "car",
                 "road", "building", "sky"]


def pick_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


class Seg3DPipeline:
    def __init__(self, device=None, depth_mode="relative"):
        self.device = pick_device(device)
        self.det_proc = AutoProcessor.from_pretrained(DET_ID)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(DET_ID).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(SAM_ID)
        self.sam = SamModel.from_pretrained(SAM_ID).to(self.device).eval()
        self._dep_cache, self.clip, self.timings = {}, None, {}
        self.use_depth(depth_mode)

    def use_depth(self, mode):
        if mode not in self._dep_cache:
            pid = DEPTH_IDS[mode]
            self._dep_cache[mode] = (AutoImageProcessor.from_pretrained(pid),
                                     AutoModelForDepthEstimation.from_pretrained(pid).to(self.device).eval())
        self.dep_proc, self.dep = self._dep_cache[mode]
        self.depth_mode, self.metric = mode, mode in ("indoor", "outdoor")

    def get_clip(self):
        if self.clip is None:
            from semantic import ClipLifter
            self.clip = ClipLifter(self.device)
        return self.clip

    def _to_dev(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                v = v.float() if v.dtype == torch.float64 else v   # MPS has no float64
                v = v.to(self.device)
            out[k] = v
        return out

    def _timed(self, name, fn, *a):
        sync(self.device); t0 = time.perf_counter()
        out = fn(*a)
        sync(self.device); self.timings[name] = (time.perf_counter() - t0) * 1000
        return out

    # ---- stage: text -> boxes (Grounding DINO)
    @torch.no_grad()
    def detect(self, image, prompt, box_th, text_th):
        text = ". ".join(p.strip().lower() for p in prompt.split(",") if p.strip()) + "."
        inputs = self._to_dev(self.det_proc(images=image, text=text, return_tensors="pt"))
        out = self.det(**inputs)
        r = self.det_proc.post_process_grounded_object_detection(
            out, inputs["input_ids"], threshold=box_th, text_threshold=text_th,
            target_sizes=[image.size[::-1]])[0]
        labels = r.get("text_labels", r["labels"])
        return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), [str(l) for l in labels]

    def _sam_forward(self, inputs):
        sizes = (inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        feed = self._to_dev({k: v for k, v in inputs.items() if k not in ("original_sizes", "reshaped_input_sizes")})
        out = self.sam(**feed, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(out.pred_masks.cpu(), *sizes)[0]
        return masks.reshape(-1, *masks.shape[-2:]).numpy().astype(bool), out.iou_scores.cpu().reshape(-1).numpy()

    # ---- stage: boxes -> masks (SAM)
    @torch.no_grad()
    def segment(self, image, boxes):
        if len(boxes) == 0:
            return np.zeros((0, image.size[1], image.size[0]), bool)
        inputs = self.sam_proc(image, input_boxes=[boxes.tolist()], return_tensors="pt")
        return self._sam_forward(inputs)[0]

    # ---- stage: grid of point prompts -> class-agnostic masks (SAM, 'segment everything')
    @torch.no_grad()
    def segment_grid(self, image, grid=6, min_iou=0.85):
        W, H = image.size
        pts = [(float(x), float(y)) for y in (np.arange(grid) + 0.5) / grid * H for x in (np.arange(grid) + 0.5) / grid * W]
        try:
            m, s = self._sam_forward(self.sam_proc(image, input_points=[[[list(p)] for p in pts]], return_tensors="pt"))
        except Exception:      # fallback: one prompt at a time
            ms, ss = [], []
            for p in pts:
                a, b = self._sam_forward(self.sam_proc(image, input_points=[[list(p)]], return_tensors="pt"))
                ms.append(a[0]); ss.append(b[0])
            m, s = np.stack(ms), np.array(ss)
        area = m.reshape(len(m), -1).mean(1)
        ok = (s >= min_iou) & (area > 0.003) & (area < 0.6)
        m, s = m[ok], s[ok]
        if len(m) == 0:
            return m, s
        keep = nms_masks(m, s)
        return m[keep], s[keep]

    # ---- stage: depth (Depth Anything V2)
    @torch.no_grad()
    def estimate_depth(self, image):
        inputs = self._to_dev(self.dep_proc(images=image, return_tensors="pt"))
        d = self.dep(**inputs).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None], size=image.size[::-1], mode="bicubic", align_corners=False)
        return d[0, 0].float().cpu().numpy()

    # ---- stage: lift masks to 3D
    def lift(self, rgb, raw, boxes, scores, labels, masks, sources, gt_depth, K):
        K = default_K(rgb.shape[1], rgb.shape[0]) if K is None else K
        depth = np.clip(raw, 0.1, 80.0) if self.metric else disparity_to_depth(raw, gt_depth)
        xyz = backproject(depth, K)
        id_map = np.zeros(depth.shape, np.int32); insts = []
        for b, s, l, m, src in zip(boxes, scores, labels, masks, sources):
            if m.sum() < 100:
                continue
            pts, col = xyz[m], rgb[m]
            keep, box3d = trimmed_box(pts)
            pts, col = pts[keep], col[keep]
            if len(pts) < 30:
                continue
            insts.append(Instance(l, float(s), np.asarray(b, float), m, pts, col, box3d, obb_9dof(pts), None, src))
            id_map[m] = len(insts)
        return depth, insts, xyz, id_map, K

    # ---- full scene build
    def build(self, image, vocab="", box_th=0.35, text_th=0.25, use_free=True, grid=6, use_clip=True,
              gt_depth=None, K=None):
        image = image.convert("RGB"); rgb = np.array(image); self.timings = {}
        if vocab.strip():
            boxes, scores, labels = self._timed("detect", self.detect, image, vocab, box_th, text_th)
            dmasks = self._timed("segment", self.segment, image, boxes)
        else:
            boxes, scores, labels, dmasks = np.zeros((0, 4)), np.zeros(0), [], np.zeros((0,) + rgb.shape[:2], bool)
        masks, sc, lb, bx, src = list(dmasks), list(scores), list(labels), list(boxes), ["detector"] * len(dmasks)
        if use_free:
            fm, fs = self._timed("segment_free", self.segment_grid, image, grid)
            if len(fm):
                iou = mask_iou_matrix(fm, np.asarray(dmasks)) if len(dmasks) else np.zeros((len(fm), 1))
                for i in range(len(fm)):
                    if iou[i].max() < 0.5:
                        masks.append(fm[i]); sc.append(float(fs[i])); lb.append("object"); bx.append(mask_box(fm[i])); src.append("free")
        raw = self._timed("depth", self.estimate_depth, image)
        depth, insts, xyz, id_map, K = self._timed("lift", self.lift, rgb, raw, bx, sc, lb, masks, src, gt_depth, K)
        if use_clip and insts:
            clip = self.get_clip()
            embs = self._timed("clip", clip.embed_masks, rgb, [i.mask for i in insts])
            for i, e in zip(insts, embs):
                i.emb = e
            free = [k for k, i in enumerate(insts) if i.source == "free"]
            if free:
                extra = [w.strip().lower() for w in vocab.split(",") if w.strip()]
                names, conf = clip.zero_shot(embs[free], list(dict.fromkeys(extra + DEFAULT_VOCAB)))
                for k, n in zip(free, names):
                    insts[k].label = n
        self.timings["total"] = sum(self.timings.values())
        return {"rgb": rgb, "depth": depth, "instances": insts, "xyz": xyz, "id_map": id_map, "K": K,
                "metric": self.metric, "timings": dict(self.timings)}

    def run(self, image, prompt, box_th=0.35, text_th=0.25, gt_depth=None, K=None):
        """Detector-only mode (used by evaluate.py)."""
        return self.build(image, prompt, box_th, text_th, use_free=False, use_clip=False, gt_depth=gt_depth, K=K)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--vocab", default="chair, table, sofa, lamp, person")
    ap.add_argument("--depth", default="relative", choices=list(DEPTH_IDS)); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    sc = Seg3DPipeline(a.device, a.depth).build(Image.open(a.image), a.vocab)
    draw_overlay(sc["rgb"], sc["instances"]).save("results/overlay.png")
    save_scene_ply("results/scene.ply", sc["xyz"], sc["rgb"], sc["id_map"])
    print([(i.label, i.source) for i in sc["instances"]]); print(sc["timings"])
