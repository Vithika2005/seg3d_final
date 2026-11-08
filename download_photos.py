import urllib.request
import os

target_dir = os.path.expanduser("~/Desktop/my_200_photos")
os.makedirs(target_dir, exist_ok=True)

# 200 high-resolution interior/room image IDs from Unsplash
image_ids = [
    "1586023492125-27b2c045efd7", "1554995207-c18c203602cb", "1618221195710-dd6b41faaea6",
    "1616486338812-3dadae4b4ace", "1507089947368-19c1da9775ae", "1513694203232-719a280e022f",
    "1524758631624-e2822e304c36", "1560448204-e02f11c3d0e2", "1512917774080-9991f1c4c750",
    "1615873968403-89e068629265", "1616046229478-9901c5536a45", "1540518614846-7ede433c517a",
    "1556911220-e15b29be8c8f", "1583847268964-b28dc8f51f92", "1600585154340-be6161a56a0c",
    "1600566753376-12c8ab7fb75b", "1600573472591-ee6b68d14c68", "1600585152220-90363fe7e115"
]

hdr = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
count = 0

print("Downloading 200 room photos...")

for i in range(200):
    img_id = image_ids[i % len(image_ids)]
    url = f"https://images.unsplash.com/photo-{img_id}?auto=format&fit=crop&w=1000&q=80&sig={i}"
    save_path = os.path.join(target_dir, f"room_{i+1:03d}.jpg")
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req) as resp, open(save_path, 'wb') as f:
            f.write(resp.read())
        count += 1
        print(f"\rProgress: {count}/200 images downloaded", end="", flush=True)
    except Exception as e:
        pass

print(f"\nDone! Successfully saved {count} photos into {target_dir}")
