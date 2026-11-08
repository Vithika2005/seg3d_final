"""Gradio demo: build a 3D scene from one image, ask language/spatial questions, fly a splat camera."""
import json, os
import numpy as np
import gradio as gr
from PIL import Image

from geom3d import (camera_path, draw_overlay, fill_holes, render_splats, save_scene_ply, scene_to_json, PALETTE)
from reasoning import SceneReasoner, singular
from seg3d import Seg3DPipeline
from viz import save_video, scene_fig

os.makedirs("results", exist_ok=True)
_pipe = None
EXAMPLES = ["the chair to the left of the table", "what is on top of the table", "closest lamp",
            "how many chairs", "distance between the sofa and the table", "the largest pillow",
            "something you can sit on", "the object behind the sofa", "what is between the sofa and the window"]


def pipe():
    global _pipe
    if _pipe is None:
        _pipe = Seg3DPipeline()
    return _pipe


def make_scorer(scene, mode="combined"):
    insts, clip = scene["instances"], pipe().get_clip()
    embs = np.stack([i.emb for i in insts])
    heads = [set(singular(w) for w in i.label.split()) for i in insts]

    def scorer(phrase):
        rel = clip.relevancy(embs, phrase)
        head = singular(phrase.split()[-1]) if phrase.split() else ""
        lab = np.array([1.0 if head in h else 0.0 for h in heads])
        if mode == "clip":
            return rel
        if mode == "label":
            return lab
        return np.maximum(rel, lab * (0.5 + 0.5 * rel))     # label match boosts CLIP relevancy
    return scorer


def rows(scene):
    out = []
    for k, i in enumerate(scene["instances"]):
        c = (i.box3d[0] + i.box3d[1]) / 2; s = i.box3d[1] - i.box3d[0]
        out.append([k, i.label, i.source, round(i.score, 2), *np.round(c, 2).tolist(), *np.round(s, 2).tolist(), len(i.points)])
    return out


def f_build(image, vocab, depth_mode, box_th, text_th, use_free, grid, max_pts, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    p = pipe(); p.use_depth(depth_mode)
    progress(0.2, desc="detect + segment + depth + CLIP (first run downloads models)")
    sc = p.build(Image.fromarray(image), vocab, box_th, text_th, use_free, int(grid))
    if not sc["instances"]:
        raise gr.Error("No objects found - lower the thresholds or edit the vocabulary.")
    ply = save_scene_ply("results/scene.ply", sc["xyz"], sc["rgb"], sc["id_map"])
    js = "results/scene.json"; json.dump(scene_to_json(sc), open(js, "w"), indent=2)
    info = f"**device** {p.device} | **depth** {depth_mode} | " + " | ".join(f"{k}: {v:.0f} ms" for k, v in sc["timings"].items())
    return (sc, draw_overlay(sc["rgb"], sc["instances"]), scene_fig(sc, int(max_pts), "instance"), rows(sc),
            ply, js, info, {"t": [], "a": []})


def f_recolor(scene, mode, max_pts):
    if scene is None:
        raise gr.Error("Build a scene first.")
    return scene_fig(scene, int(max_pts), mode)


def f_ask(scene, query, max_pts):
    if scene is None:
        raise gr.Error("Build a scene first (tab 1).")
    r = SceneReasoner(scene["instances"], make_scorer(scene), metric=scene["metric"]).ask(query)
    t, a = r["targets"], r["anchors"]
    md = "### " + r["text"] + "\n\n" + "\n".join(f"- {s}" for s in r["steps"])
    ov = draw_overlay(scene["rgb"], scene["instances"], highlight=t, anchors=a) if (t or a) else Image.fromarray(scene["rgb"])
    return md, ov, scene_fig(scene, int(max_pts), "rgb", t, a), {"t": t, "a": a}


def f_splat(scene, qs, path, n, amp, quality, hl, progress=gr.Progress()):
    if scene is None:
        raise gr.Error("Build a scene first.")
    rgb = scene["rgb"]; H, W = rgb.shape[:2]; insts = scene["instances"]
    col = rgb.copy().reshape(-1, 3).astype(np.float32)
    if hl and (qs["t"] or qs["a"]):
        col *= 0.4
        for k in qs["t"] + qs["a"]:
            tint = np.array([255, 70, 70] if k in qs["t"] else [60, 200, 255], np.float32)
            m = insts[k].mask.reshape(-1); col[m] = 0.45 * rgb.reshape(-1, 3)[m] + 0.55 * tint
    col = col.clip(0, 255).astype(np.uint8)
    xyz = scene["xyz"]; s = 2 if quality.startswith("fast") else 1
    X = xyz[::s, ::s].reshape(-1, 3); C = col.reshape(H, W, 3)[::s, ::s].reshape(-1, 3)
    ok = np.isfinite(X).all(1); X, C = X[ok], C[ok]
    pivot = np.array([0, 0, float(np.median(X[:, 2]))])
    frames = []
    for k, (R, c) in enumerate(camera_path(path, pivot, int(n), float(amp))):
        progress(k / int(n), desc=f"splatting frame {k + 1}/{int(n)}")
        img, cov = render_splats(X, C, scene["K"], R, c, (W, H), radius=1, sigma=0.75 if s == 1 else 1.1)
        frames.append(fill_holes(img, cov)[0])
    return save_video(frames, "results/flythrough.mp4", fps=12)


with gr.Blocks(title="Language-Guided 3D Scene Understanding") as demo:
    scene_state, q_state = gr.State(None), gr.State({"t": [], "a": []})
    gr.Markdown("# Language-guided 3D instance segmentation, retrieval & splat view synthesis\n"
                "Grounding DINO + SAM -> Depth Anything V2 -> 3D lift -> CLIP features -> spatial reasoning -> Gaussian-splat rendering")
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(type="numpy", label="Input image")
            vocab = gr.Textbox("chair, table, sofa, bed, lamp, cabinet, desk, door, window, pillow, plant, tv, bookshelf",
                               label="Detector vocabulary (comma separated; may be empty)")
            depth_mode = gr.Dropdown(["relative", "indoor", "outdoor"], value="relative", label="Depth model (indoor/outdoor = metric)")
            with gr.Accordion("Advanced", open=False):
                box_th = gr.Slider(0.1, 0.8, 0.35, label="Box threshold"); text_th = gr.Slider(0.1, 0.8, 0.25, label="Text threshold")
                use_free = gr.Checkbox(True, label="Also segment everything with SAM (open-vocabulary objects)")
                grid = gr.Slider(4, 10, 6, step=1, label="SAM prompt grid (n x n)")
                max_pts = gr.Slider(10000, 100000, 50000, step=5000, label="Max 3D points drawn")
            build_btn = gr.Button("Build 3D scene", variant="primary")
            info = gr.Markdown()
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("1. Scene"):
                    ov = gr.Image(label="2D instance masks")
                    color_mode = gr.Radio(["instance", "rgb"], value="instance", label="Point colour")
                    plot = gr.Plot(label="Rotatable 3D scene")
                    table = gr.Dataframe(headers=["id", "label", "source", "score", "cx", "cy", "cz", "sx", "sy", "sz", "#pts"])
                with gr.Tab("2. Ask the scene"):
                    q = gr.Textbox(label="Language / spatial query", placeholder=EXAMPLES[0])
                    gr.Examples(EXAMPLES, q)
                    ask_btn = gr.Button("Ask", variant="primary")
                    ans = gr.Markdown(); ov_q = gr.Image(label="Highlighted (red = result, cyan = anchor)"); plot_q = gr.Plot()
                with gr.Tab("3. Novel views (Gaussian splats)"):
                    path = gr.Radio(["orbit", "dolly", "sway"], value="orbit", label="Camera path")
                    n = gr.Slider(12, 72, 36, step=6, label="Frames"); amp = gr.Slider(5, 30, 15, label="Orbit angle (deg)")
                    quality = gr.Radio(["fast (half-res points)", "full"], value="fast (half-res points)", label="Quality")
                    hl = gr.Checkbox(True, label="Highlight last query result")
                    splat_btn = gr.Button("Render fly-through", variant="primary"); vid = gr.Video(label="Splat view synthesis")
                with gr.Tab("4. Export"):
                    ply = gr.File(label="Coloured point cloud with instance ids (.ply)"); js = gr.File(label="Instance metadata (.json)")
    build_btn.click(f_build, [img, vocab, depth_mode, box_th, text_th, use_free, grid, max_pts],
                    [scene_state, ov, plot, table, ply, js, info, q_state])
    color_mode.change(f_recolor, [scene_state, color_mode, max_pts], plot)
    ask_btn.click(f_ask, [scene_state, q, max_pts], [ans, ov_q, plot_q, q_state])
    q.submit(f_ask, [scene_state, q, max_pts], [ans, ov_q, plot_q, q_state])
    splat_btn.click(f_splat, [scene_state, q_state, path, n, amp, quality, hl], vid)

if __name__ == "__main__":
    demo.launch()
