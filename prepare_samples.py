"""Copy your room photos into samples/ : fixes phone rotation, resizes (long side 1024), names them room1.jpg, room2.jpg ...
Usage: python prepare_samples.py "/path/to/photos" [output_folder]   (default output: samples/; jpg/jpeg/png; export HEIC to JPG first)"""
import os, sys
from PIL import Image, ImageOps

src = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "samples"
os.makedirs(out_dir, exist_ok=True)
files = sorted(os.path.join(src, f) for f in os.listdir(src) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"})
assert files, "no jpg/png files found in that folder"
for i, f in enumerate(files, 1):
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    im.thumbnail((1024, 1024))
    out = f"{out_dir}/room{i}.jpg"; im.save(out, quality=92)
    print(f"{f} -> {out} {im.size}")
