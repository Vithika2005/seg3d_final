"""Build queries.json interactively (no hand-editing JSON).
For each image in samples/: type your queries in the terminal (blank line = done), then click the correct object for each.
Run prepare_samples.py FIRST so clicks match the resized images. Re-running appends for images not yet in queries.json.
Usage: python make_queries.py"""
import glob, json, os
import matplotlib.pyplot as plt
from PIL import Image

OUT = "queries.json"
data = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {d["image"] for d in data}
plt.ion(); quit_now = False
for path in sorted(glob.glob("samples/*.jpg")):
    if path in done or quit_now:
        continue
    im = Image.open(path)
    if max(im.size) > 1100:
        print(f"WARNING {path} is {im.size}: run prepare_samples.py first, otherwise clicks will not match.")
    plt.clf(); plt.imshow(im); plt.title(path); plt.draw(); plt.pause(0.3)
    print(f"\n=== {path} ===   (blank line = next image, q = save and quit; start a query with ! if its target is NOT in your detector vocabulary, e.g. !the red mug)")
    qs = []
    while True:
        s = input("query: ").strip()
        if s == "q":
            quit_now = True
        if s in ("", "q"):
            break
        qs.append(s)
    for s in qs:
        plt.title(f"CLICK the correct object for: {s.lstrip(chr(33))}"); plt.draw()
        x, y = plt.ginput(1, timeout=0)[0]
        data.append({"image": path, "query": s.lstrip("!").strip(), "click": [int(x), int(y)], "tag": "oov" if s.startswith("!") else "in_vocab"})
        json.dump(data, open(OUT, "w"), indent=2)
print(f"saved {len(data)} queries to {OUT}")
