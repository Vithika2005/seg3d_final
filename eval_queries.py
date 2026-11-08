"""Evaluate language-guided retrieval + spatial reasoning on your own labelled queries.
queries.json: [{"image": "samples/room1.jpg", "query": "the chair left of the table", "click": [x, y], "tag": "oov"}, ...]
'click' = a pixel inside the correct object; optional 'tag' = "oov" if the target is NOT in --vocab.
Writes results/queries_<scorer>[_nofree].json and prints a table.
Ablations: --scorer {combined,clip,label}   --no-free (disable SAM-grid free segments)"""
import argparse, json, collections
from PIL import Image
from app import make_scorer, pipe
from reasoning import SceneReasoner, parse
from stats_util import summarize

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--vocab", default="chair, table, sofa, bed, lamp, cabinet, desk, door, window, pillow, plant, tv, bookshelf")
ap.add_argument("--depth", default="relative")
ap.add_argument("--scorer", default="combined", choices=["combined", "clip", "label"])
ap.add_argument("--no-free", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()
Q = json.load(open(a.file)); p = pipe(); p.use_depth(a.depth)
cache = {}
by_type, by_tag, tot = collections.defaultdict(lambda: [0, 0, 0, 0]), collections.defaultdict(lambda: [0, 0, 0, 0]), [0, 0, 0, 0]
for q in Q:
    if q["image"] not in cache:
        cache[q["image"]] = p.build(Image.open(q["image"]), a.vocab, use_free=not a.no_free)
    sc = cache[q["image"]]
    r = SceneReasoner(sc["instances"], make_scorer(sc, a.scorer), metric=sc["metric"]).ask(q["query"])
    x, y = q["click"]
    hit = [bool(sc["instances"][t].mask[y, x]) for t in r["targets"]]
    row = [1, int(bool(hit and hit[0])), int(any(hit)), len(r["targets"])]
    for acc in (by_type[parse(q["query"])["kind"]], by_tag[q.get("tag", "in_vocab")], tot):
        for i in range(4):
            acc[i] += row[i]
res = {"scorer": a.scorer, "free_segments": not a.no_free, "depth": a.depth, "overall": summarize(tot),
       "per_type": {k: summarize(v) for k, v in by_type.items()}, "per_tag": {k: summarize(v) for k, v in by_tag.items()}}
out = a.out or f"results/queries_{a.scorer}{'_nofree' if a.no_free else ''}.json"
json.dump(res, open(out, "w"), indent=2)
print(f"scorer={a.scorer} free_segments={not a.no_free} depth={a.depth}  -> {out}")
print(f"{'group':16s} n   top1   (95% CI)        any-hit  avg-returned")
for name, d in [*res["per_type"].items(), *[("tag:" + k, v) for k, v in res["per_tag"].items()], ("ALL", res["overall"])]:
    print(f"{name:16s} {d['n']:<3d} {d['top1']:.2f}   ({d['top1_ci95'][0]:.2f}-{d['top1_ci95'][1]:.2f})   {d['any_hit']:.2f}     {d['avg_returned']:.1f}")
