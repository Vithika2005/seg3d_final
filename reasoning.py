"""Language-guided retrieval + 3D spatial reasoning over lifted instances.
Camera frame: x right, y down, z forward. 'left/right' = as seen from the camera."""
import re
import numpy as np
from geom3d import aabb_gap

STOP = {"the", "a", "an", "that", "this", "my", "please", "find", "show", "me", "highlight", "locate",
        "where", "is", "are", "which", "what", "select", "get", "all", "of", "any", "there", "us"}
REL = [("to the left of", "left"), ("on the left of", "left"), ("left of", "left"),
       ("to the right of", "right"), ("on the right of", "right"), ("right of", "right"),
       ("in front of", "front"), ("behind", "behind"), ("in back of", "behind"),
       ("on top of", "on"), ("on", "on"), ("above", "above"), ("over", "above"),
       ("underneath", "below"), ("beneath", "below"), ("under", "below"), ("below", "below"),
       ("next to", "near"), ("close to", "near"), ("near", "near"), ("beside", "near"),
       ("between", "between")]
REL.sort(key=lambda x: -len(x[0]))
SUP = {"closest": ("dist", "min"), "nearest": ("dist", "min"), "farthest": ("dist", "max"),
       "furthest": ("dist", "max"), "leftmost": ("x", "min"), "rightmost": ("x", "max"),
       "highest": ("y", "min"), "topmost": ("y", "min"), "lowest": ("y", "max"),
       "largest": ("vol", "max"), "biggest": ("vol", "max"), "smallest": ("vol", "min")}
GENERIC = {"", "object", "objects", "thing", "things", "item", "items", "stuff", "something", "anything"}


def singular(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "ches")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def clean(p):
    return " ".join(w for w in re.findall(r"[a-z0-9\-']+", p.lower()) if w not in STOP)


def parse(q):
    s = q.lower().strip().rstrip("?.! ")
    m = re.search(r"how many (.+)", s)
    if m:
        return {"kind": "count", "target": clean(m.group(1))}
    m = re.search(r"(?:distance|how far).*?between (.+?) and (.+)", s) or \
        re.search(r"how far (?:away )?(?:is|are) (.+?) (?:from|to) (.+)", s) or \
        re.search(r"distance (?:from|of) (.+?) to (.+)", s)
    if m:
        return {"kind": "distance", "a": clean(m.group(1)), "b": clean(m.group(2))}
    m = re.search(r"how far (?:away )?(?:is|are) (.+)", s) or re.search(r"distance to (.+)", s)
    if m:
        return {"kind": "camera_distance", "target": clean(m.group(1))}
    m = re.search(r"\b(" + "|".join(SUP) + r")\b(.*)$", s)
    if m:
        before, after = clean(s[:m.start()]), m.group(2).strip()
        mm = re.match(r"(?:to|from)\s+(.+)", after)
        anchor = None
        if mm:
            target, anchor = before or "object", clean(mm.group(1))
            if anchor in {"camera", "viewer", "view"}:
                anchor = None
        else:
            target = clean(after) or before or "object"
        return {"kind": "superlative", "sup": m.group(1), "target": target, "anchor": anchor}
    for phrase, rel in REL:
        m = re.search(r"\b" + re.escape(phrase) + r"\b", s)
        if m:
            tgt, rest = clean(s[:m.start()]), s[m.end():]
            if rel == "between":
                mm = re.match(r"\s*(.+?)\s+and\s+(.+)", rest)
                if mm:
                    return {"kind": "between", "target": tgt or "object", "a": clean(mm.group(1)), "b": clean(mm.group(2))}
                continue
            anchor = clean(rest)
            if anchor:
                return {"kind": "relation", "rel": rel, "target": tgt or "object", "anchor": anchor}
    return {"kind": "retrieve", "target": clean(s)}


class SceneReasoner:
    def __init__(self, insts, scorer, thr=0.55, rel_cut=0.85, near_thr=0.8, margin=0.15, metric=False):
        self.insts, self.scorer, self.thr, self.rel_cut = insts, scorer, thr, rel_cut
        self.near_thr, self.margin, self.metric = near_thr, margin, metric
        self.unit = "m" if metric else "units (relative depth: scale is approximate)"

    # -- helpers
    def c(self, i):
        b = self.insts[i].box3d
        return (b[0] + b[1]) / 2

    def name(self, i):
        return f"#{i} {self.insts[i].label}"

    def where(self, i):
        c = self.c(i)
        return f"{self.name(i)} (x={c[0]:+.2f}, depth={c[2]:.2f})"

    def match(self, phrase, fallback=True):
        """Instances matching a free-text phrase. Returns (indices sorted by score, confident, scores)."""
        n = len(self.insts)
        if phrase in GENERIC:
            return list(range(n)), True, np.ones(n)
        sc = np.asarray(self.scorer(phrase), float)
        if sc.size == 0:
            return [], False, sc
        best = sc.max()
        if best >= self.thr:
            cut = max(self.thr, best * self.rel_cut)
            return [int(i) for i in np.argsort(-sc) if sc[i] >= cut], True, sc
        return ([int(np.argmax(sc))] if fallback else []), False, sc

    def holds(self, rel, t, a):
        ct, ca, m = self.c(t), self.c(a), self.margin
        bt, ba = self.insts[t].box3d, self.insts[a].box3d
        if rel == "left":   return ct[0] < ca[0] - m
        if rel == "right":  return ct[0] > ca[0] + m
        if rel == "front":  return ct[2] < ca[2] - m
        if rel == "behind": return ct[2] > ca[2] + m
        if rel == "above":  return ct[1] < ca[1] - m
        if rel == "below":  return ct[1] > ca[1] + m
        if rel == "near":   return aabb_gap(bt, ba) < self.near_thr
        if rel == "on":
            touching = abs(bt[1][1] - ba[0][1]) < 0.3                       # target bottom ~ anchor top (y is down)
            overlap = all(min(bt[1][k], ba[1][k]) - max(bt[0][k], ba[0][k]) > -0.1 for k in (0, 2))
            return bool(touching and overlap)
        return False

    def _result(self, text, targets, anchors, steps):
        return {"text": text, "targets": list(targets), "anchors": list(anchors), "steps": steps}

    # -- main entry
    def ask(self, q):
        if not self.insts:
            return self._result("The scene has no instances.", [], [], [])
        p = parse(q); k = p["kind"]; steps = [f"Parsed query as **{k}**: `{ {a: b for a, b in p.items() if a != 'kind'} }`"]

        if k == "count":
            idx, _, _ = self.match(p["target"], fallback=False)
            return self._result(f"I count **{len(idx)}** matching '{p['target']}'" + (": " + ", ".join(self.name(i) for i in idx) if idx else "."), idx, [], steps)

        if k == "distance":
            ia, ib = self.match(p["a"])[0][0], self.match(p["b"])[0][0]
            d = float(np.linalg.norm(self.c(ia) - self.c(ib)))
            return self._result(f"{self.name(ia)} to {self.name(ib)}: **{d:.2f} {self.unit}** (centre to centre).", [ia], [ib], steps)

        if k == "camera_distance":
            i = self.match(p["target"])[0][0]
            return self._result(f"{self.name(i)} is **{np.linalg.norm(self.c(i)):.2f} {self.unit}** from the camera.", [i], [], steps)

        if k == "superlative":
            metric, how = SUP[p["sup"]]
            cand, _, _ = self.match(p["target"])
            anc = self.match(p["anchor"])[0] if p["anchor"] else []
            cand = [i for i in cand if i not in anc]
            if not cand:
                return self._result("No candidate objects found.", [], anc, steps)

            def val(i):
                if metric == "dist":
                    if anc:
                        return min(float(np.linalg.norm(self.c(i) - self.c(a))) for a in anc)
                    return float(np.linalg.norm(self.c(i)))
                if metric == "x": return float(self.c(i)[0])
                if metric == "y": return float(self.c(i)[1])
                return float(np.prod(self.insts[i].box3d[1] - self.insts[i].box3d[0]))
            best = (min if how == "min" else max)(cand, key=val)
            steps.append(f"Ranked {len(cand)} candidate(s) by {metric}; picked {self.name(best)}.")
            return self._result(f"The {p['sup']} {p['target']}: {self.where(best)}.", [best], anc, steps)

        if k == "between":
            ia, ib = self.match(p["a"])[0][0], self.match(p["b"])[0][0]
            ca, cb = self.c(ia), self.c(ib); ab = cb - ca; L2 = float(ab @ ab) + 1e-9
            cand, _, _ = self.match(p["target"])
            hits = []
            for t in cand:
                if t in (ia, ib):
                    continue
                s = float((self.c(t) - ca) @ ab) / L2
                dist = float(np.linalg.norm(self.c(t) - (ca + s * ab)))
                if 0.1 <= s <= 0.9 and dist < max(0.5, 0.4 * np.sqrt(L2)):
                    hits.append(t)
            txt = f"Between {self.name(ia)} and {self.name(ib)}: " + (", ".join(self.where(t) for t in hits) if hits else "nothing found.")
            return self._result(txt, hits, [ia, ib], steps)

        if k == "relation":
            cand, _, _ = self.match(p["target"])
            anc, conf, _ = self.match(p["anchor"])
            cand = [t for t in cand if t not in anc]
            hits = [t for t in cand if any(self.holds(p["rel"], t, a) for a in anc)]
            steps.append(f"Anchor '{p['anchor']}' -> {', '.join(self.name(a) for a in anc)}; tested `{p['rel']}` on {len(cand)} candidate(s) in 3D.")
            if hits:
                return self._result(f"{len(hits)} match(es) ({p['rel']} {p['anchor']}): " + "; ".join(self.where(t) for t in hits), hits, anc, steps)
            return self._result(f"No '{p['target']}' found {p['rel']} '{p['anchor']}'.", [], anc, steps)

        idx, conf, sc = self.match(p["target"])
        steps.append("Retrieval = detector label match + CLIP relevancy (LERF-style, vs canonical phrases).")
        tag = "" if conf else " (low confidence - best guess)"
        return self._result(f"Found {len(idx)}{tag}: " + "; ".join(self.where(i) for i in idx), idx, [], steps)
