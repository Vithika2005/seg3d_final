## Experiment 1 - NYU Depth V2 (Depth Anything V2 Small, relative depth aligned to GT scale/shift; detector-only)

| box_th | AbsRel (lower better) | delta1 | mask mIoU | recall@0.5 | precision@0.5 | 3D box IoU (matched) | GT instances |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.067 | 0.949 | 0.491 | 0.571 | 0.233 | 0.379 | 487 |
| 0.35 | 0.067 | 0.949 | 0.395 | 0.468 | 0.332 | 0.390 | 487 |
| 0.45 | 0.067 | 0.949 | 0.272 | 0.322 | 0.430 | 0.402 | 487 |

Note: precision undercounts because NYU labels are not exhaustive; 3D IoU is axis-aligned and uses GT-aligned depth.

## Depth backbone (box_th 0.35)

| Backbone | AbsRel | delta1 | 3D box IoU | depth stage (ms/img) | total (ms/img) |
|---|---|---|---|---|---|
| Depth Anything V2 Small | 0.067 | 0.949 | 0.390 | 47 | 1606 |
| Depth Anything V2 Base | 0.063 | 0.952 | 0.419 | 142 | 1892 |

Latency measured during evaluation on device `mps`, first image excluded (warm-up).

## Experiment 2a - Retrieval / spatial reasoning per query type (combined scorer)

| Query type | n | top-1 (95% CI) | any-hit | avg objects returned |
|---|---|---|---|---|
| count | 4 | 0.75 (0.30-0.95) | 0.75 | 1.5 |
| retrieve | 18 | 0.50 (0.29-0.71) | 0.56 | 1.3 |
| relation | 2 | 0.50 (0.10-0.91) | 0.50 | 1.0 |
| superlative | 2 | 0.50 (0.10-0.91) | 0.50 | 1.0 |
| between | 1 | 0.00 (0.00-0.79) | 1.00 | 5.0 |
| ALL | 27 | 0.52 (0.34-0.69) | 0.59 | 1.4 |

## Experiment 2b - Ablation

| Variant | n | top-1 (95% CI) | any-hit | avg returned | out-of-vocab top-1 |
|---|---|---|---|---|---|
| label match only | 27 | 0.33 (0.19-0.52) | 0.44 | 1.4 | - |
| CLIP relevancy only | 27 | 0.48 (0.31-0.66) | 0.52 | 1.4 | - |
| combined (default) | 27 | 0.52 (0.34-0.69) | 0.59 | 1.4 | - |
| combined, no free segments | 27 | 0.52 (0.34-0.69) | 0.52 | 1.2 | - |
## Custom Roboflow dataset (COCO-style AP, same metric code for both methods)

| Method | box AP50 | box AP | mask AP50 | mask AP | mask P@0.5 | mask R@0.5 | ms / image |
|---|---|---|---|---|---|---|---|
| Ours (zero-shot) | 0.347 | 0.256 | 0.345 | 0.232 | 0.538 | 0.219 | 1462 |
| YOLO-seg (trained on our train split) | 0.001 | 0.000 | 0.001 | 0.000 | 0.006 | 0.028 | 88 |

