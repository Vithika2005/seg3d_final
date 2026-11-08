"""Model-free unit tests: python tests/test_core.py   (needs numpy, pillow, plotly only)"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geom3d import (Instance, backproject, box_corners, camera_path, default_K, fill_holes, obb_9dof,
                    project_points, render_splats, save_scene_ply, scene_to_json, nms_masks)
from reasoning import SceneReasoner, parse
from cocoeval_lite import evaluate_predictions, polygons_to_mask


def test_backproject_roundtrip():
    K = default_K(64, 48); d = np.full((48, 64), 3.0); P = backproject(d, K)
    uv = project_points(P.reshape(-1, 3), K).reshape(48, 64, 2)
    assert np.allclose(uv[..., 0], np.arange(64)[None]) and np.allclose(uv[..., 1], np.arange(48)[:, None], atol=1e-6)


def test_splat_identity():
    H, W = 96, 128; K = default_K(W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx * 2 % 256, yy * 2 % 256, (xx + yy) % 256], -1).astype(np.uint8)
    xyz = backproject(np.full((H, W), 3.0), K).reshape(-1, 3)
    R, c = camera_path("orbit", [0, 0, 3], 12)[0]
    out, cov = render_splats(xyz, img.reshape(-1, 3), K, R, c, (W, H))
    out, cov = fill_holes(out, cov)
    err = np.abs(out.astype(float) - img.astype(float))[2:-2, 2:-2].mean()
    assert cov.mean() > 0.99 and err < 25, (cov.mean(), err)     # gradients wrap at 256 -> loose bound
    # a moved camera must change the image
    R2, c2 = camera_path("orbit", [0, 0, 3], 12, amp=20)[3]
    out2, _ = render_splats(xyz, img.reshape(-1, 3), K, R2, c2, (W, H))
    assert np.abs(out2.astype(float) - out.astype(float)).mean() > 1


def test_obb():
    rs = np.random.RandomState(0); th = 0.6
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    pts = (rs.uniform(-0.5, 0.5, (4000, 3)) * np.array([4.0, 1.5, 1.7])) @ R.T + np.array([2, 1, 15])
    o = obb_9dof(pts)
    assert np.allclose(sorted(o["size"]), [1.5, 1.7, 4.0], atol=0.3), o["size"]
    assert np.allclose(o["center"], [2, 1, 15], atol=0.15)
    assert box_corners(o["center"], o["size"], o["R"]).shape == (8, 3)


def fake_scene():
    def inst(label, lo, hi):
        lo, hi = np.array(lo, float), np.array(hi, float)
        return Instance(label, 1.0, np.zeros(4), np.zeros((4, 4), bool), np.random.rand(50, 3) * (hi - lo) + lo,
                        np.zeros((50, 3), np.uint8), np.stack([lo, hi]))
    # camera frame: x right, y down (so smaller y = higher), z forward
    return [inst("table", [-0.5, 0.4, 2.5], [0.5, 0.9, 3.5]),      # 0
            inst("chair", [-1.8, 0.4, 2.8], [-1.2, 1.0, 3.2]),     # 1 left of table
            inst("chair", [1.2, 0.4, 2.8], [1.8, 1.0, 3.2]),       # 2 right of table
            inst("lamp", [-0.1, 0.0, 2.9], [0.1, 0.4, 3.1]),       # 3 on top of table
            inst("sofa", [-0.8, 0.3, 4.5], [0.8, 1.0, 5.5])]       # 4 behind table


def scorer_for(insts):
    return lambda phrase: np.array([1.0 if phrase.split()[-1].rstrip("s") in i.label else 0.0 for i in insts])


def test_parser():
    assert parse("how many chairs are there")["kind"] == "count"
    assert parse("the chair to the left of the table") == {"kind": "relation", "rel": "left", "target": "chair", "anchor": "table"}
    assert parse("closest lamp")["kind"] == "superlative"
    assert parse("distance between the sofa and the table")["kind"] == "distance"
    assert parse("what is between the sofa and the table")["kind"] == "between"
    assert parse("find the red mug")["kind"] == "retrieve"


def test_reasoning():
    ins = fake_scene(); R = SceneReasoner(ins, scorer_for(ins))
    assert R.ask("the chair to the left of the table")["targets"] == [1]
    assert R.ask("the chair to the right of the table")["targets"] == [2]
    assert R.ask("what is on top of the table")["targets"] == [3]
    assert R.ask("how many chairs")["targets"] == [1, 2]
    assert R.ask("the object behind the table")["targets"] == [4]
    assert R.ask("closest lamp")["targets"] == [3]
    assert R.ask("the farthest sofa")["targets"] == [4]
    assert R.ask("the largest table")["targets"] == [0]
    assert R.ask("the leftmost chair")["targets"] == [1]
    assert R.ask("the chair closest to the sofa")["targets"][0] in (1, 2)
    d = R.ask("distance between the lamp and the sofa")
    assert d["targets"] == [3] and d["anchors"] == [4] and "m" in d["text"] or "units" in d["text"]
    assert R.ask("how far is the table")["targets"] == [0]


def test_export_and_fig():
    ins = fake_scene(); H, W = 8, 8
    xyz = np.random.rand(H, W, 3) + [0, 0, 1]; idm = np.zeros((H, W), np.int32); idm[:2] = 1
    with tempfile.TemporaryDirectory() as d:
        p = save_scene_ply(os.path.join(d, "a.ply"), xyz, np.zeros((H, W, 3), np.uint8), idm, stride=1)
        raw = open(p, "rb").read(); hdr = raw.split(b"end_header\n")[0]
        assert b"element vertex 64" in hdr and len(raw) == len(hdr) + len(b"end_header\n") + 64 * 19
    for i in ins:
        i.obb = obb_9dof(i.points)
    sc = {"instances": ins, "rgb": np.zeros((H, W, 3), np.uint8), "xyz": xyz, "id_map": idm, "metric": False}
    assert len(scene_to_json(sc)["instances"]) == 5
    try:
        from viz import scene_fig
        assert len(scene_fig(sc, 1000).data) > 5 and len(scene_fig(sc, 1000, "rgb", [1], [0]).data) > 5
    except ImportError:
        print("plotly missing - figure test skipped")


def test_ap():
    def sq(x0, y0, x1, y1):
        m = np.zeros((20, 20), bool); m[y0:y1, x0:x1] = True; return m, (x0, y0, x1, y1)
    gm1, gb1 = sq(2, 2, 10, 10); gm2, gb2 = sq(5, 5, 15, 15)
    gts = {"a": {1: [{"box": gb1, "mask": gm1}], 2: [{"box": gb2, "mask": gm2}]}}
    perfect = [{"img": 1, "cls": "a", "score": 0.9, "box": gb1, "mask": gm1}, {"img": 2, "cls": "a", "score": 0.8, "box": gb2, "mask": gm2},
               {"img": 1, "cls": "a", "score": 0.1, "box": gb2, "mask": gm2}]       # low-score false positive after full recall
    r = evaluate_predictions(perfect, gts)
    assert abs(r["mask"]["AP"] - 1.0) < 1e-9 and abs(r["box"]["AP50"] - 1.0) < 1e-9 and r["mask"]["recall50"] == 1.0
    wm, wb = sq(15, 15, 19, 19)
    bad = [{"img": 1, "cls": "a", "score": 0.9, "box": wb, "mask": wm}]
    assert evaluate_predictions(bad, gts)["mask"]["AP50"] == 0.0
    half = [{"img": 1, "cls": "a", "score": 0.9, "box": gb1, "mask": gm1}]
    assert abs(evaluate_predictions(half, gts)["mask"]["AP50"] - 0.505) < 0.02       # recall 0.5 at precision 1 -> ~0.505 (101-pt)
    assert polygons_to_mask([[2, 2, 10, 2, 10, 10, 2, 10]], 20, 20).sum() > 50


def test_nms():
    a = np.zeros((3, 20, 20), bool); a[0, :10] = True; a[1, :10] = True; a[2, 15:] = True
    assert sorted(nms_masks(a, [0.9, 0.8, 0.7])) == [0, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
