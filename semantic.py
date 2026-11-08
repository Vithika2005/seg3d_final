"""CLIP features lifted onto 3D instances + LERF-style relevancy scoring."""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"
CANON = ("object", "things", "stuff", "texture")      # LERF canonical phrases


class ClipLifter:
    def __init__(self, device):
        self.device = device
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self._tcache = {}

    @staticmethod
    def _norm(f):
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    @torch.no_grad()
    def embed_images(self, pil_list, bs=16):
        out = []
        for i in range(0, len(pil_list), bs):
            pix = self.proc(images=pil_list[i:i + bs], return_tensors="pt")["pixel_values"].to(self.device)
            out.append(self._norm(self.model.get_image_features(pixel_values=pix)))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    @torch.no_grad()
    def embed_text(self, texts):
        key = tuple(texts)
        if key not in self._tcache:
            inp = self.proc(text=list(texts), return_tensors="pt", padding=True).to(self.device)
            self._tcache[key] = self._norm(self.model.get_text_features(**inp))
        return self._tcache[key]

    def embed_masks(self, rgb, masks, pad=0.15):
        """Crop each mask's bounding box (background greyed out), pad to square, embed with CLIP."""
        H, W = rgb.shape[:2]; crops = []
        for m in masks:
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
            y0, y1, x0, x1 = max(0, y0 - py), min(H, y1 + py), max(0, x0 - px), min(W, x1 + px)
            sub = rgb[y0:y1, x0:x1].copy(); sub[~m[y0:y1, x0:x1]] = 128
            h, w = sub.shape[:2]; s = max(h, w)
            canvas = np.full((s, s, 3), 128, np.uint8)
            canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = sub
            crops.append(Image.fromarray(canvas))
        return self.embed_images(crops)

    def relevancy(self, emb, query, temp=100.0):
        """LERF relevancy: min over canonical phrases of pairwise softmax(query vs canonical)."""
        t = self.embed_text([query] + list(CANON))
        s = temp * emb @ t.T
        q, c = s[:, :1], s[:, 1:]
        return (1.0 / (1.0 + np.exp(np.clip(c - q, -50, 50)))).min(1)

    def zero_shot(self, emb, vocab, temp=100.0):
        t = self.embed_text([f"a photo of a {v}" for v in vocab])
        lg = temp * emb @ t.T; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        return [vocab[i] for i in p.argmax(1)], p.max(1)
