import pickle
from pathlib import Path

cache_path = Path("output_results/v2_detections.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

print("Checking items in left desk region (x: 800~1000, y: 200~400):")
for f_idx in sorted(cache.keys()):
    if f_idx % 24 != 0 and f_idx != max(cache.keys()):
        continue
    p, items = cache[f_idx]
    left_items = [it for it in items if 800 <= it["bbox"][0] <= 1100 and 200 <= it["bbox"][1] <= 500]
    if left_items:
        names = [f"{it['cname']} ({int(it['bbox'][0])},{int(it['bbox'][1])})" for it in left_items]
        print(f"Frame {f_idx}: {', '.join(names)}")
