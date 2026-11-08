"""Shows an image; click the CORRECT object for your query; prints the click in the format queries.json needs.
Usage: python click_point.py samples/room1.jpg"""
import sys
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open(sys.argv[1])
if max(im.size) > 1100:
    print(f"WARNING: image is {im.size}. Run prepare_samples.py first, otherwise your clicks will not match the resized image.")
plt.imshow(im); plt.title("Click inside the correct object (one click)")
x, y = plt.ginput(1, timeout=0)[0]
print(f'"click": [{int(x)}, {int(y)}]')
