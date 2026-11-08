"""Plotly 3D figures and video writing (no torch needed)."""
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from geom3d import BOX_EDGES, PALETTE, aabb_corners, box_corners


def _col(k):
    return "rgb(%d,%d,%d)" % tuple(int(x) for x in PALETTE[k % len(PALETTE)])


def _P(a):                       # camera (x right, y down, z fwd) -> plot (x, depth, up)
    a = np.asarray(a); return a[..., 0], a[..., 2], -a[..., 1]


def _rgb_str(c):
    return ["rgb(%d,%d,%d)" % (r, g, b) for r, g, b in c]


def _box_trace(corners, color, name=None):
    xs, ys, zs = [], [], []
    for i, j in BOX_EDGES:
        for c in (i, j):
            x, y, z = _P(corners[c]); xs.append(x); ys.append(y); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=5), showlegend=False, hoverinfo="skip")


def scene_fig(scene, max_pts=40000, mode="instance", targets=None, anchors=None, boxes=True, obb=True):
    """mode: 'instance' (colour = object identity) or 'rgb' (true colour). targets/anchors highlight query results."""
    insts, rgb, xyz, idm = scene["instances"], scene["rgb"], scene["xyz"], scene["id_map"]
    tg, an = set(targets or []), set(anchors or []); active = bool(tg or an)
    rs = np.random.RandomState(0); fig = go.Figure()
    bg = (idm == 0) & np.isfinite(xyz).all(-1)
    B, Bc = xyz[bg], rgb[bg]
    sel = rs.permutation(len(B))[: max_pts // 3]
    if len(sel):
        x, y, z = _P(B[sel])
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", name="background", hoverinfo="skip",
                                   marker=dict(size=1.3, color=_rgb_str(Bc[sel]), opacity=0.12 if active else 0.55)))
    per = (max_pts * 2 // 3) // max(1, len(insts))
    for k, ins in enumerate(insts):
        sel = rs.permutation(len(ins.points))[:per]; x, y, z = _P(ins.points[sel])
        if k in tg:
            color, size, op = "rgb(255,59,59)", 3.0, 1.0
        elif k in an:
            color, size, op = "rgb(60,200,255)", 3.0, 1.0
        elif active:
            color, size, op = _rgb_str(ins.colors[sel]), 1.4, 0.10
        elif mode == "instance":
            color, size, op = _col(k), 2.2, 1.0
        else:
            color, size, op = _rgb_str(ins.colors[sel]), 2.0, 1.0
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", name=f"#{k} {ins.label}",
                                   marker=dict(size=size, color=color, opacity=op)))
        show_box = (k in tg or k in an) if active else (boxes and ins.source == "detector")
        if show_box:
            c = box_corners(ins.obb["center"], ins.obb["size"], ins.obb["R"]) if (obb and ins.obb is not None) else aabb_corners(ins.box3d)
            fig.add_trace(_box_trace(c, color if isinstance(color, str) else _col(k)))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="X (right)", yaxis_title="Depth Z", zaxis_title="Up",
                                 camera=dict(eye=dict(x=0.0, y=-1.7, z=0.45), up=dict(x=0, y=0, z=1))),
                      margin=dict(l=0, r=0, t=0, b=0), height=600, showlegend=len(insts) <= 25)
    return fig


def save_video(frames, path, fps=12):
    """mp4 via imageio-ffmpeg; falls back to GIF. Frames cropped to even size for H.264."""
    frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception:
        gif = path.rsplit(".", 1)[0] + ".gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        return gif
