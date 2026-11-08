# Custom dataset with Roboflow (200 images) and metrics

Goal: build your own instance-segmentation dataset, then report detection + segmentation metrics for (a) this zero-shot pipeline and (b) a YOLO-seg model trained on your data. Both are scored by the same code (`evaluate_coco.py`), so the comparison is fair.

## 1. Collect 200 images
- Indoor photos you took yourself: different rooms, lighting, angles, clutter levels. Aim for 6-10 object classes that appear often (for example chair, table, sofa, bed, lamp, pillow, plant, tv).
- Use **single lowercase words** as class names (`chair`, not `Chair_01`): the pipeline's detector is prompted with them.
- Resize to about 1024 px on the long side and fix phone rotation (this does not touch `samples/`):
```bash
python prepare_samples.py "/path/to/your/200/photos" dataset_raw
```
- Privacy: avoid faces, screens with personal content and documents. Roboflow free plans typically make projects public, so check the visibility settings before uploading.

## 2. Annotate in Roboflow (the interface changes over time; names may differ slightly)
1. Create a free account and a **new project of type "Instance Segmentation"** (not "Object Detection": the metrics need polygon masks).
2. Upload `dataset_raw/`.
3. Annotate with the polygon tool, or the AI-assisted **Smart Polygon** (SAM-based) to go faster. Label **every** visible instance of your classes in every image. Missing labels lower precision.
4. Keep class names identical and lowercase.

## 3. Generate a version and export
- *Generate version*: split train/valid/test, for example 60/20/20 (about 120/40/40 images). A test set of only 20 images gives unstable metrics; 40 is better. Add **Auto-Orient**; no resizing is needed. Augmentation applies only to train and is not needed for the zero-shot pipeline.
- *Export dataset* twice and unzip:
  - format **COCO** -> `roboflow_coco/` (contains `train/`, `valid/`, `test/`, each with `_annotations.coco.json`)
  - format **YOLOv8** (or YOLO11) -> `roboflow_yolo/` (contains `data.yaml`)

## 4. Metrics for this pipeline (no training)
```bash
python evaluate_coco.py --root roboflow_coco --split test --out results/roboflow_ours.json
```
It prints box and mask AP50, AP75, AP (mean over IoU 0.50-0.95), precision and recall at IoU 0.5, and milliseconds per image. For a larger sample you may also run `--split train` and `--split valid` (nothing is trained, so no leakage), but only the `test` numbers are comparable with the YOLO baseline.

## 5. Baseline trained on your data (optional but recommended: gives you training curves)
```bash
pip install ultralytics          # if this fails on your Python version, create a venv with Python 3.12 for training only
yolo segment train data=roboflow_yolo/data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640 batch=8 device=mps
python evaluate_coco.py --root roboflow_coco --split test --yolo runs/segment/train/weights/best.pt --out results/roboflow_yolo.json
```
Training curves and Ultralytics' own validation metrics are in `runs/segment/train/` (`results.png`). If `yolo11n-seg.pt` is not found, use `yolov8n-seg.pt`.

## 6. Tables
```bash
python make_tables.py        # adds the Roboflow table to results/tables.md
```

## What to say in the report
- Dataset: number of images and instances per class, split sizes, annotation tool and how many people annotated.
- Metrics: COCO-style AP computed by our own implementation (101-point interpolation, IoU 0.50:0.05:0.95), not pycocotools, so numbers can differ slightly from other tools.
- The zero-shot pipeline is evaluated at a fixed detector threshold (`--box-th`, default 0.25); the YOLO model at confidence 0.001, the usual convention for AP.
- Expect the trained YOLO model to win on your classes if your test set resembles your training set; the point of the comparison is zero-shot flexibility (any text query, 3D output) versus supervised accuracy. Report whatever you measure.
