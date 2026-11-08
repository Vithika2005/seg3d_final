# GUIDE - Language-Guided 3D Instance Segmentation, Retrieval and Splat View Synthesis

## 1. Requirement -> implementation map
| Requirement | Where | How |
|---|---|---|
| Detect and segment objects, **instance masks not boxes** | `seg3d.py` `detect`, `segment`, `segment_grid` | Grounding DINO boxes -> SAM masks; plus SAM point-grid "segment everything" (NMS-deduplicated) so objects outside the typed vocabulary are also instances |
| Recover spatial structure / depth | `estimate_depth` | Depth Anything V2 (relative, or metric indoor/outdoor variants) |
| Lift pixels to 3D, **coloured point cloud with object identities** | `lift`, `geom3d.save_scene_ply` | Pinhole back-projection; per-instance points, 2-98 percentile outlier trim, AABB + 9-DoF PCA box; PLY has an `instance` field |
| **Rotatable 3D representation** | `viz.scene_fig` | Plotly Scatter3d (instance colours or RGB), boxes, in browser |
| **Camera view synthesis using Gaussian splatting** | `geom3d.render_splats`, `camera_path` | One Gaussian per pixel from depth + colour, soft depth-sorted splat rasteriser, orbit/dolly/sway paths -> mp4 |
| **Language-guided retrieval** | `semantic.py`, `reasoning.match` | Detector/zero-shot labels + CLIP features of each segment, LERF-style relevancy against canonical phrases |
| **Spatial reasoning** | `reasoning.py` | Parser + geometric predicates on 3D boxes (left/right/front/behind/above/below/on/near/between, closest/farthest/leftmost/.../largest, counting, distances) |
| **Highlight in reconstructed 3D scene** | `scene_fig(targets, anchors)`, `draw_overlay`, splat video | Result red, anchor cyan, rest dimmed |

## 2. Papers and what is (and is not) taken from each
Verify every citation on IEEE Xplore / CVF Open Access before submission.

| Paper | Venue | What we use | What we do NOT do |
|---|---|---|---|
| Kirillov et al., *Segment Anything* | ICCV 2023 (IEEE/CVF) | Box and point prompts -> masks | no SAM fine-tuning |
| Yang et al., *Depth Anything* (V1) | CVPR 2024 (IEEE/CVF) | Monocular depth (we run the V2 checkpoints, NeurIPS 2024) | no training |
| Szymanowicz et al., *Splatter Image: Ultra-Fast Single-View 3D Reconstruction* (verified: CVPR 2024, pp. 10208-10217, doi 10.1109/CVPR52733.2024.00972) | IEEE/CVF | The **representation**: one 3D Gaussian per input pixel (position along the camera ray from depth, colour, footprint) | Their *learned U-Net* that predicts Gaussians is **not** used: the released models are trained on object-centric data (ShapeNet, Objaverse, CO3D) and rely on a CUDA rasteriser. We set the Gaussian parameters analytically from monocular depth and render with our own NumPy rasteriser |
| Kerr et al., *LERF: Language Embedded Radiance Fields* | ICCV 2023 (IEEE/CVF) | The **relevancy score** (query vs canonical phrases, pairwise softmax, min) and the idea of lifting CLIP features into 3D | no NeRF, no multi-view optimisation; CLIP features are per segment, not a continuous field |
| Nguyen et al., *Open3DIS* | CVPR 2024 (IEEE/CVF) | Paradigm: 2D foundation-model masks + CLIP lifted to 3D instances | works on 3D scans; we use one view |
| Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* | ACM TOG 2023 (not IEEE) | Background for splat rendering | no optimisation, no anisotropic covariance, no SH colour |
| Liu et al., *Grounding DINO* | ECCV 2024 (Springer) | Text-prompted detector (component) | - |
| Radford et al., *CLIP* | ICML 2021 | Vision-language features (component) | - |

Positioning for the IEEE-style paper: *a training-free, single-image framework that composes 2D foundation models into a queryable 3D scene with splat-based view synthesis, evaluated on NYU Depth V2 and a hand-labelled retrieval benchmark, with an Apple-Silicon performance study.* State clearly that it is an integration and evaluation contribution built on pretrained models.

## 3. Step by step: what happens when you press "Build 3D scene"
1. **Detection (`detect`)**: vocabulary `chair, table` becomes `chair. table.`; Grounding DINO fuses text and image features and returns boxes whose tokens match the words (`box_th` filters box score, `text_th` filters word match).
2. **Instance masks (`segment`)**: each box is a SAM prompt -> one binary mask. This is what turns boxes into pixel-accurate instances.
3. **Segment everything (`segment_grid`)**: an n x n grid of point prompts gives class-agnostic masks; masks with predicted IoU < 0.85, tiny or huge masks are dropped, near duplicates removed by mask-IoU NMS, and any mask overlapping a detector mask (IoU >= 0.5) is discarded. Remaining ones become `source="free"` instances so *anything* can be queried.
4. **Depth (`estimate_depth`)**: Depth Anything V2 predicts relative disparity. Without ground truth it is mapped to an assumed 0.5-8 m range (sizes are approximate); with a metric model (`indoor`/`outdoor`) output is in metres; in evaluation we fit scale and shift to the Kinect depth in inverse-depth space.
5. **Lifting (`lift`)**: `X=(u-cx)Z/fx`, `Y=(v-cy)Z/fy`, `Z=depth` (60 degree FOV assumed if no calibration). Pixels of each mask become that object's points; the 2-98 percentile trim removes edge "flying pixels"; AABB and PCA 9-DoF boxes are computed; `id_map` stores identities per pixel.
6. **CLIP features (`embed_masks`)**: each mask's bounding box is cropped, the background greyed, padded to a square and embedded. Free segments get a name by zero-shot classification against ~45 common indoor/outdoor nouns plus your vocabulary.
7. **Figure and export**: Plotly 3D scene; PLY with identities; JSON with metadata.

## 4. Language-guided retrieval and spatial reasoning
**Scoring a phrase.** `score = max(CLIP relevancy, label_match * (0.5 + 0.5 * relevancy))`. Relevancy is LERF's: for each canonical phrase c in {object, things, stuff, texture}, `sigmoid(100*(q.e - c.e))`, take the minimum. Candidates are instances with score above 0.55 and within 85 % of the best score; if none, the best instance is returned and flagged "low confidence".

**Query grammar** (`reasoning.parse`), checked in this order: `how many X` -> count; `distance between X and Y` / `how far is X from Y` -> Euclidean distance; `how far is X` -> distance to camera; superlatives `closest|farthest|leftmost|rightmost|highest|lowest|largest|smallest X [to Y]`; `X between Y and Z`; relations `X left of / right of / in front of / behind / on top of / above / below / near Y`; otherwise plain retrieval (open vocabulary).

**Geometry** (camera frame: x right, y down, z forward; "left/right" as seen from the camera): left `cx_t < cx_a - 0.15`; front `cz_t < cz_a - 0.15`; above `cy_t < cy_a - 0.15` (y is down); on top: target bottom within 0.3 of anchor top and footprints overlap; near: AABB gap < 0.8; between: projection on segment A-B in [0.1, 0.9] and perpendicular distance < max(0.5, 0.4|AB|). Margins are in scene units: tune `SceneReasoner(margin=..., near_thr=...)` when using relative depth.

**Limits to state in the report**: the parser is rule-based (no free-form language understanding); one view means hidden objects cannot be retrieved; relative depth makes distances approximate.

## 5. Gaussian-splat view synthesis (`geom3d.render_splats`)
Each pixel is lifted to a Gaussian with centre `xyz`, colour `rgb`, and an isotropic footprint (sigma 0.75 px). For a novel pose `(R, c)`: `p_cam = R (p - c)`, project with K, add each Gaussian to the 3x3 pixels around its projection with weight `exp(-d^2 / 2 sigma^2)`, keep only fragments within 4 % of the nearest depth per pixel (visibility), and normalise the weighted colour sum. Small holes are filled from neighbours. Cameras: `orbit` (yaw +-amp deg around a pivot at median depth), `dolly`, `sway`. Cost is dominated by one `argsort` per frame (about 0.9 s at 640x480 on a generic CPU, 0.2 s with half-res points, measured in testing without a GPU). Hardware note: official 3DGS/Splatter Image rasterisers are CUDA-only and do not run on macOS; this implementation is CPU/NumPy so it runs on any Mac, and the *model* stages run on MPS. Large camera moves reveal regions never seen (single view) - they appear as holes; this is inherent and belongs in the limitations slide.

## 6. Experiments (each gives a table or figure)
1. **Detection/segmentation/depth/3D box** on NYU Depth V2 (`evaluate.py`): depth AbsRel and delta1, mask mIoU, recall and precision at IoU 0.5, 3D box IoU of matched instances. Run `--depth-mode relative` and `indoor` and compare (masks do not change between depth modes, only depth and 3D IoU do), and sweep `--box-th 0.25 0.35 0.45` for the precision/recall trade-off.
2. **Retrieval and spatial reasoning** (`eval_queries.py`): 30-50 own labelled queries; report top-1 and any-hit accuracy per query type. Ablate with `--scorer label|clip|combined`.
3. **Free segments on/off** (`eval_queries.py --no-free`, or the UI checkbox): effect on retrieval of out-of-vocabulary objects (use queries whose target is NOT in the vocabulary).
4. **Depth backbone**: `python evaluate.py ... --depth-mode relative_base` (Small vs Base); AbsRel / 3D IoU vs the `depth` entry of `mean_ms` in the JSON.
5. **Hardware** (`benchmark.py`): CPU vs MPS per stage, memory, splat render time.
6. **Failure modes**: mirrors and glass, low light, motion blur, heavy occlusion, tiny objects, left/right ambiguity. Two screenshots each.

## 7. PPT outline (15 slides) and demo
1 Title. 2 Motivation (robots/drones need 3D, language interface). 3 Problem statement and objectives. 4 Related work (SAM, Depth Anything, LERF, Splatter Image, Open3DIS). 5 Architecture diagram (top of README). 6 Detection + instance masks. 7 Depth and 3D lifting (equations). 8 Point cloud with identities (screenshot, PLY). 9 CLIP lifting and LERF relevancy. 10 Spatial reasoning (grammar and predicates, 3 worked examples). 11 Gaussian-splat view synthesis (equations, fly-through GIF/video). 12 Results tables (NYU + retrieval). 13 Hardware benchmark chart. 14 Failure modes and limitations. 15 Conclusion, future work (multi-view/video fusion, tracking, learned Gaussian predictor, metric depth), references.
**Live demo (3 min)**: build a room image; rotate 3D; ask `the chair to the left of the table`, then `something you can sit on`, then `distance between the sofa and the table`; render the orbit fly-through with the result highlighted. Pre-build the scene and keep a recorded video as backup (model load takes time).

## 8. Likely viva questions
- *Where is the deep learning, since you did not train?* Four pretrained deep networks (Grounding DINO, SAM, Depth Anything, CLIP) are composed; the research contributions are the 3D lifting, the language/spatial reasoning layer, the splat renderer, and the evaluation.
- *Why not use the Splatter Image network?* It is trained for single objects and needs CUDA; for room scenes we derive Gaussians from monocular depth. Using their checkpoints on object crops on a cloud GPU is a possible extension.
- *How is this different from LERF?* LERF optimises a NeRF over many views (minutes to hours) with a dense CLIP field; we are feed-forward on one image (seconds) with per-segment CLIP features, and keep LERF's relevancy score.
- *Is the distance in metres?* Only with a metric depth model; otherwise approximate.
- *What if two tables exist?* All candidates within 85 % of the best score are anchors; the relation holds if it is true for any anchor.
- *Why is the rendered view broken at wide angles?* Single-view disocclusion: no data for hidden surfaces.

## 9. Report skeleton (IEEE two-column)
Abstract; I Introduction; II Related Work; III Method (pipeline, lifting equations, relevancy, reasoning predicates, splat rendering); IV Experimental Setup (NYU Depth V2, own query set, Apple M5, PyTorch MPS); V Results and Ablations; VI Failure Analysis; VII Conclusion; References.

## 10. Honest status
- Verified in the build environment: all model-free logic passes `tests/test_core.py` (back-projection round trip, splat render at identity pose reproduces the image, 9-DoF box recovery, parser, spatial predicates, PLY writing, Plotly figure). The splat timings above come from that environment.
- **Not verified**: anything that needs the model weights (Grounding DINO, SAM, Depth Anything, CLIP) or an Apple GPU, because they were not available where the code was written. Expect small fixes on first run (most likely a `transformers` argument or an MPS fallback). Send the traceback.
- Not included: video tracking, occupancy/blind-spot prediction, KITTI monocular 9-DoF benchmarking, CoreML export, learned Gaussian prediction, NeRF/3DGS optimisation. `obb` gives 9-DoF PCA boxes per instance but they come from visible surface points only.
- All metrics in your report must come from your own runs.
