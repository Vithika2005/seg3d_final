"""Pure-numpy geometry, Gaussian-splat rendering and export. No torch needed (unit-testable)."""
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PALETTE = np.random.RandomState(3).randint(40, 255, (64, 3))


@dataclass
class Instance:
    label: str
    score: float
    box2d: np.ndarray      # (4,) x0,y0,x1,y1
    mask: np.ndarray       # (H,W) bool
    points: np.ndarray     # (N,3) camera-frame XYZ (x right, y down, z forward)
    colors: np.ndarray     # (N,3) uint8
    box3d: np.ndarray      # (2,3) axis-aligned min,max
    obb: Any = None        # 9-DoF oriented box dict
    emb: Any = None        # CLIP embedding (D,)
    source: str = "detector"   # "detector" (text-prompted) or "free" (SAM grid)


# ---------------- camera / depth ----------------
def default_K(w, h, fov_deg=60.0):
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def backproject(depth, K):
    """Pinhole model: X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=depth. Returns (H,W,3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)


def project_points(P, K):
    uv = np.asarray(P) @ K.T
    return uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)


def disparity_to_depth(disp, gt=None, near=0.5, far=8.0):
    """Relative disparity (bigger=closer) -> depth. With GT: scale+shift fit in inverse-depth space.
    Without GT: mapped to an assumed [near, far] metre range (approximate)."""
    if gt is not None:
        m = gt > 1e-3
        A = np.stack([disp[m], np.ones(m.sum())], 1)
        a, b = np.linalg.lstsq(A, 1.0 / gt[m], rcond=None)[0]
        return np.clip(1.0 / np.clip(a * disp + b, 1e-3, None), 0.1, 10.0)
    dn = (disp - disp.min()) / (np.ptp(disp) + 1e-8)
    return 1.0 / (1 / far + dn * (1 / near - 1 / far))


# ---------------- boxes / masks ----------------
def trimmed_box(pts, q=2):
    lo, hi = np.percentile(pts, [q, 100 - q], axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), 1)
    return keep, np.stack([pts[keep].min(0), pts[keep].max(0)])


def box3d_iou(a, b):
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    inter = np.prod(np.clip(hi - lo, 0, None))
    va, vb = np.prod(a[1] - a[0]), np.prod(b[1] - b[0])
    return float(inter / (va + vb - inter + 1e-9))


def aabb_gap(a, b):
    d = np.maximum(0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(d))


def mask_box(m):
    ys, xs = np.where(m)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_iou_matrix(A, B, ds=4):
    a = np.asarray(A)[:, ::ds, ::ds].reshape(len(A), -1).astype(np.float32)
    b = np.asarray(B)[:, ::ds, ::ds].reshape(len(B), -1).astype(np.float32)
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter + 1e-6)


def nms_masks(masks, scores, thr=0.7):
    order = np.argsort(-np.asarray(scores)); keep = []
    iou = mask_iou_matrix(masks, masks)
    for i in order:
        if all(iou[i, j] <= thr for j in keep):
            keep.append(int(i))
    return keep


def euler_zyx_from_R(R):
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])])
    return np.degrees([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0])


def obb_9dof(pts, q=2):
    """PCA oriented box: 3 translation + 3 size + 3 rotation (Euler) = 9 DoF."""
    c0 = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - c0).T))
    R = vecs[:, ::-1].copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    local = (pts - c0) @ R
    lo, hi = np.percentile(local, [q, 100 - q], axis=0)
    return {"center": c0 + R @ ((lo + hi) / 2), "size": hi - lo, "R": R, "euler_deg": euler_zyx_from_R(R)}


_SIGNS = np.array(list(itertools.product([-1, 1], repeat=3)), float)
BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def box_corners(center, size, R):
    return np.asarray(center) + (_SIGNS * np.asarray(size) / 2) @ np.asarray(R).T


def aabb_corners(b):
    return box_corners((b[0] + b[1]) / 2, b[1] - b[0], np.eye(3))


# ---------------- Gaussian-splat view synthesis ----------------
def look_at(c, target):
    zf = np.asarray(target, float) - c
    zf = zf / np.linalg.norm(zf)
    x = np.cross([0, 1, 0], zf); x /= np.linalg.norm(x)
    y = np.cross(zf, x)
    return np.stack([x, y, zf]), np.asarray(c, float)   # p_cam = R @ (p - c)


def camera_path(kind, pivot, n=36, amp=18.0):
    pivot = np.asarray(pivot, float); poses = []
    for k in range(n):
        s = np.sin(2 * np.pi * k / n)
        if kind == "orbit":
            th = np.radians(amp) * s
            Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
            c = pivot + Ry @ (-pivot)
        elif kind == "dolly":
            c = np.array([0, 0, 0.45 * pivot[2] * (0.5 - 0.5 * np.cos(2 * np.pi * k / n))])
        else:  # sway
            c = np.array([0.25 * pivot[2] * s, 0, 0])
        poses.append(look_at(c, pivot))
    return poses


def render_splats(xyz, rgb, K, R, c, size, radius=1, sigma=0.75, tol=0.04):
    """One Gaussian per pixel (Splatter-Image-style representation), rendered with a
    depth-sorted, Gaussian-weighted soft z-buffer. xyz (N,3) float, rgb (N,3) uint8, size=(W,H).
    Returns (image uint8 HxWx3, coverage bool HxW)."""
    W, H = size
    p = (xyz - c) @ R.T
    z = p[:, 2]; ok = z > 0.1
    p, col, z = p[ok], rgb[ok].astype(np.float32), z[ok]
    u = K[0, 0] * p[:, 0] / z + K[0, 2]; v = K[1, 1] * p[:, 1] / z + K[1, 2]
    idx = np.arange(len(z)); F, Z, I, Wt = [], [], [], []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            pu = np.round(u).astype(np.int32) + dx; pv = np.round(v).astype(np.int32) + dy
            keep = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
            w = np.exp(-((pu - u) ** 2 + (pv - v) ** 2) / (2 * sigma ** 2))
            F.append((pv * W + pu)[keep]); Z.append(z[keep]); I.append(idx[keep]); Wt.append(w[keep])
    F, Z, I, Wt = map(np.concatenate, (F, Z, I, Wt))
    img = np.zeros((H * W, 3), np.float32)
    if len(F) == 0:
        return img.reshape(H, W, 3).astype(np.uint8), np.zeros((H, W), bool)
    order = np.argsort(Z, kind="stable")
    uniq, first = np.unique(F[order], return_index=True)
    zmin = np.full(H * W, np.inf, np.float32); zmin[uniq] = Z[order][first]
    k = Z <= zmin[F] * (1 + tol)
    f, w, ci = F[k], Wt[k], I[k]
    ws = np.bincount(f, weights=w, minlength=H * W)
    for ch in range(3):
        img[:, ch] = np.bincount(f, weights=w * col[ci, ch], minlength=H * W)
    m = ws > 1e-3
    img[m] /= ws[m, None]
    return img.reshape(H, W, 3).clip(0, 255).astype(np.uint8), m.reshape(H, W)


def fill_holes(img, cov, iters=3):
    img = img.astype(np.float32); cov = cov.copy()
    for _ in range(iters):
        acc = np.zeros_like(img); cnt = np.zeros(cov.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc += np.roll(np.roll(img * cov[..., None], dy, 0), dx, 1)
                cnt += np.roll(np.roll(cov.astype(np.float32), dy, 0), dx, 1)
        new = (~cov) & (cnt >= 3)
        img[new] = acc[new] / cnt[new, None]; cov = cov | new
    return img.clip(0, 255).astype(np.uint8), cov


# ---------------- export / drawing ----------------
def save_scene_ply(path, xyz, rgb, id_map, stride=2):
    """Binary PLY: x y z red green blue instance (0 = background, k+1 = k-th instance)."""
    P = xyz[::stride, ::stride].reshape(-1, 3); C = rgb[::stride, ::stride].reshape(-1, 3)
    I = id_map[::stride, ::stride].reshape(-1)
    ok = np.isfinite(P).all(1) & (P[:, 2] > 0)
    arr = np.empty(int(ok.sum()), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"),
                                          ("green", "u1"), ("blue", "u1"), ("instance", "<i4")])
    arr["x"], arr["y"], arr["z"] = P[ok, 0], P[ok, 1], P[ok, 2]
    arr["red"], arr["green"], arr["blue"] = C[ok, 0], C[ok, 1], C[ok, 2]
    arr["instance"] = I[ok]
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property int instance\nend_header\n" % len(arr)).encode()
    with open(path, "wb") as f:
        f.write(header); arr.tofile(f)
    return path


def scene_to_json(scene):
    out = []
    for k, i in enumerate(scene["instances"]):
        d = {"instance_id": k + 1, "label": i.label, "source": i.source, "score": round(float(i.score), 3),
             "box2d": [round(float(v), 1) for v in i.box2d], "n_points": int(len(i.points)),
             "centroid": np.median(i.points, 0).round(3).tolist(),
             "aabb_min": i.box3d[0].round(3).tolist(), "aabb_max": i.box3d[1].round(3).tolist()}
        if i.obb is not None:
            d["obb"] = {"center": i.obb["center"].round(3).tolist(), "size": i.obb["size"].round(3).tolist(),
                        "euler_deg_rpy": i.obb["euler_deg"].round(1).tolist()}
        out.append(d)
    return {"metric_depth": bool(scene.get("metric")), "frame": "camera: x right, y down, z forward",
            "instances": out}


def draw_overlay(rgb, insts, highlight=None, anchors=None):
    anchors = list(anchors or [])
    out = rgb.astype(np.float32).copy()
    show = list(range(len(insts))) if highlight is None else list(highlight) + anchors
    if highlight is not None:
        out *= 0.45
    colors = {}
    for k in show:
        if highlight is None:
            colors[k] = PALETTE[k % len(PALETTE)]
        else:
            colors[k] = np.array([255, 70, 70]) if k in highlight else np.array([60, 200, 255])
        out[insts[k].mask] = 0.5 * rgb[insts[k].mask] + 0.5 * colors[k]
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    for k in show:
        c = tuple(int(x) for x in colors[k]); b = insts[k].box2d.tolist()
        d.rectangle(b, outline=c, width=3)
        d.text((b[0] + 3, b[1] + 3), f"#{k} {insts[k].label}", fill=(255, 255, 255))
    return img
