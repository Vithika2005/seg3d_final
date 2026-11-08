# RUNBOOK - run everything, step by step

Conventions
- Every command is run in **Terminal**, inside your project folder, with the virtual environment active:
  `cd ~/Desktop/seg3d_final` then `source .venv/bin/activate` (the prompt shows `(.venv)`).
- Run one thing at a time and read the output before continuing. **Never paste commands into a program that is asking you a question** (that created the junk queries earlier).
- Time plan: start the NYU download (step 5) first in its own Terminal tab, then do steps 2-4 and the Roboflow annotation (step 6) while it downloads.

## 1. Put the latest files into your project folder
```bash
cd ~/Desktop/seg3d_final
ls ~/Downloads | grep seg3d                       # note the exact zip name (it may be "seg3d_project (1).zip")
rm -rf /tmp/seg3d_new && unzip -o ~/Downloads/seg3d_project.zip -d /tmp/seg3d_new
cp -R /tmp/seg3d_new/seg3d_project/. .            # the final "." copies hidden files too; your data/, samples/, results/ stay
ls make_queries_ui.py evaluate_coco.py download_nyu.sh check_nyu.py RUNBOOK.md     # all five must be listed
source .venv/bin/activate
pip install -r requirements.txt
python tests/test_core.py                          # expect lines starting with PASS
```

## 2. Photos for the retrieval benchmark (5-10 indoor photos)
Keep your **originals in a separate folder** (for example `~/Desktop/my_photos`), never inside `samples/`.
```bash
rm -f samples/*.jpg                                # start clean
python prepare_samples.py "$HOME/Desktop/my_photos"   # writes samples/room1.jpg, room2.jpg ... (rotation fixed, max 1024 px)
ls samples                                         # only roomN.jpg files should be there
python seg3d.py samples/room1.jpg --vocab "chair, table, sofa"     # smoke test; open results/overlay.png to look at it
```
Photos should show 3 or more objects (chair, table, sofa, lamp, pillow, plant, tv ...), taken from standing height.

## 3. Build the query file with the browser tool
```bash
rm -f queries.json results/queries_*.json
python make_queries_ui.py                          # prints a link: open http://127.0.0.1:7861 in your browser
```
For every query: (1) choose the image in the dropdown, (2) type **one** query, (3) **click on the correct object** in the image (a red circle appears), (4) tick *out-of-vocabulary* if the target is not in the detector vocabulary (chair, table, sofa, bed, lamp, cabinet, desk, door, window, pillow, plant, tv, bookshelf), (5) press **Add query**. The table at the bottom grows; *Delete last* removes a mistake.
Aim for 30-40 queries over your photos, per photo about:
- 2 relation: `the chair to the left of the table`, `what is on top of the table`
- 1 superlative: `closest lamp` or `the largest pillow`
- 1 between: `what is between the sofa and the window` (only if three things line up)
- 1-2 plain: `something you can sit on`; and one out-of-vocabulary such as `the painting` or `the rug` (tick the box)
Only write queries whose correct answer is **one** clear object that is visible in that photo. Stop the tool with Ctrl+C in Terminal when done, then:
```bash
python -c "import json; print(len(json.load(open('queries.json'))), 'queries')"
```

## 4. Retrieval experiments
```bash
python eval_queries.py --file queries.json                 # combined scorer
python eval_queries.py --file queries.json --scorer clip
python eval_queries.py --file queries.json --scorer label
python eval_queries.py --file queries.json --no-free
ls results/queries_*.json                                  # four files
```
Check that the `ALL` row shows n equal to your number of queries.

## 5. NYU Depth V2 (depth, masks, 3D boxes)
Start this in its **own Terminal tab** and leave it running (about 3 GB; the time depends on your connection). Keep the file **outside Desktop and Documents**: iCloud sync can interfere with large files.
```bash
cd ~/Desktop/seg3d_final && source .venv/bin/activate
pkill curl 2>/dev/null; rm -f data/nyu_depth_v2_labeled.mat       # remove the old corrupted file
bash download_nyu.sh "$HOME/nyu_data" --fresh                       # if interrupted, run again WITHOUT --fresh to resume
```
Only one download may run at a time (the script refuses to start a second one). When it prints "Size is correct":
```bash
python check_nyu.py "$HOME/nyu_data/nyu_depth_v2_labeled.mat"      # reads every slice; must print PASS
export NYU="$HOME/nyu_data/nyu_depth_v2_labeled.mat"
python evaluate.py --mat "$NYU" --n 3 --out results/test.json       # quick test, then delete results/test.json
python evaluate.py --mat "$NYU" --n 100 --box-th 0.35 --out results/nyu_small_035.json
python evaluate.py --mat "$NYU" --n 100 --box-th 0.25 --out results/nyu_small_025.json
python evaluate.py --mat "$NYU" --n 100 --box-th 0.45 --out results/nyu_small_045.json
python evaluate.py --mat "$NYU" --n 100 --depth-mode relative_base --box-th 0.35 --out results/nyu_base_035.json
```
If `check_nyu.py` prints FAIL: run the download again with `--fresh`, use another network (phone hotspot, campus Wi-Fi), or copy a good `nyu_depth_v2_labeled.mat` (2972037809 bytes) from a classmate by USB or Drive and put it in `~/nyu_data/`. If the 3-image test is slow, use `--n 50` for the real runs.

## 6. Roboflow dataset (200 images) and its metrics
**6a. Prepare images** (this does not touch `samples/`):
```bash
python prepare_samples.py "$HOME/Desktop/my_200_photos" dataset_raw
ls dataset_raw | wc -l                                 # about 200
```
**6b. Roboflow (website; menu names can change slightly)**
1. roboflow.com -> sign up (free) -> create a workspace.
2. **Create New Project** -> name it -> **Project Type: Instance Segmentation** -> create.
3. **Upload**: drag the `dataset_raw` folder in -> *Save and Continue*.
4. **Annotate**: open an image, choose the **polygon tool** or **Smart Polygon** (AI-assisted), click the object, type its class (lowercase, one word: `chair`, `table`, `sofa`, `lamp`, `pillow`, `plant`, `tv` ...), confirm. Label **every visible instance** of your classes in **every** image, with identical class names. Use 6-10 classes. Move to the next image until all 200 are done.
5. **Add images to the dataset**: choose a split, for example 60 / 20 / 20 (train / valid / test).
6. **Generate a version**: Preprocessing -> keep **Auto-Orient**, **remove the default Resize** (it stretches images to 640x640); Augmentation -> none (optional for YOLO training). Create the version.
7. **Export dataset** -> format **COCO** -> *download zip* -> unzip into `~/Desktop/seg3d_final/roboflow_coco/` so that this file exists: `roboflow_coco/test/_annotations.coco.json`.
8. Export again -> format **YOLOv8** -> unzip into `roboflow_yolo/` (contains `data.yaml`).
**6c. Metrics for this pipeline (no training):**
```bash
ls roboflow_coco/test/_annotations.coco.json
python evaluate_coco.py --root roboflow_coco --split test --out results/roboflow_ours.json
```
**6d. Baseline trained on your data** (gives training curves; needs `ultralytics`). If `pip install ultralytics` fails on Python 3.14, use a separate environment:
```bash
brew install python@3.12
python3.12 -m venv ~/venv_train && source ~/venv_train/bin/activate
pip install ultralytics tqdm pillow numpy
```
Open `roboflow_yolo/data.yaml` and set absolute paths (Roboflow's relative paths often fail):
```yaml
path: /Users/vithikasurve/Desktop/seg3d_final/roboflow_yolo
train: train/images
val: valid/images
test: test/images
```
```bash
cd ~/Desktop/seg3d_final
yolo segment train data=roboflow_yolo/data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640 batch=8 device=mps   # use device=cpu if MPS errors; yolov8n-seg.pt if yolo11n-seg.pt is not found
python evaluate_coco.py --root roboflow_coco --split test --yolo runs/segment/train/weights/best.pt --out results/roboflow_yolo.json
```
(The training venv has no Grounding DINO; `evaluate_coco.py --yolo` does not need it.) Training curves: `runs/segment/train/results.png`.

## 7. Tables, hardware benchmark, screenshots
```bash
source .venv/bin/activate
python make_tables.py                                   # results/tables.md and results/pr_curve.png
python benchmark.py --images "samples/*.jpg"            # results/benchmark.csv and benchmark.png
python app.py                                           # http://127.0.0.1:7860
```
In the app: build a scene, take a screenshot of the 2D overlay and of the 3D view (Cmd+Shift+4), ask the demo queries, render a fly-through (saved as `results/flythrough.mp4`). Take two screenshots for each failure case: mirror or glass, dim room, blurry photo, crowded scene, tiny objects. Record the demo with Cmd+Shift+5 (screen recording) as a backup.

## 8. Fill the PowerPoint
Open `Language_Guided_3D_Scene_Understanding.pptx`: drag screenshots onto the dashed boxes (slides 6, 8, 11, 14), copy the numbers from `results/tables.md` into the dashes on slides 13 and 14, set your guide/college/date on slide 1 and `[N]` on slide 12. Or paste `results/tables.md` into the chat and ask for the filled deck.

## Expected files
`results/` should end up with: `nyu_small_025.json`, `nyu_small_035.json`, `nyu_small_045.json`, `nyu_base_035.json`, `queries_combined.json`, `queries_clip.json`, `queries_label.json`, `queries_combined_nofree.json`, `roboflow_ours.json`, `roboflow_yolo.json`, `benchmark.csv`, `benchmark.png`, `tables.md`, `pr_curve.png`, `flythrough.mp4`.

## If something fails
| Symptom | Fix |
|---|---|
| `No such file ... make_queries_ui.py` | step 1 was not completed; redo the copy command |
| `bad heap free list`, `truncated file`, `file signature not found` | NYU file damaged: `python check_nyu.py ...`, then `bash download_nyu.sh "$HOME/nyu_data" --fresh` |
| `queries.json not found` | step 3 not done; the file is created only after you press Add query |
| eval_queries shows n = 2 | `queries.json` still has the old test queries: `rm queries.json` and redo step 3 |
| `Dataset not found` in YOLO training | fix the paths in `data.yaml` (step 6d) |
| `evaluate_coco.py`: KeyError or empty classes | you exported a detection project (boxes only) or the wrong format: create an Instance Segmentation project and export COCO |
| Any other traceback | paste the last 15 lines into the chat |
