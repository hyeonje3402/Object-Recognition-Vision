import pickle
from pathlib import Path

cache_path = Path("output_results/v2_detections.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

print("Tracing bottle detections around x: 2000~2300, y: 350~650:")
for f_idx in sorted(cache.keys()):
    if f_idx % 24 != 0 and f_idx != max(cache.keys()):
        continue
    _, dets = cache[f_idx]
    bottles = [d for d in dets if d["cname"] in {"bottle", "cup"} and 2000 <= d["bbox"][0] <= 2300 and 350 <= d["bbox"][1] <= 650]
    b_strs = [f"{d['cname']} {[int(x) for x in d['bbox']]}" for d in bottles]
    if b_strs:
        print(f"Frame {f_idx:3d}: {', '.join(b_strs)}")
    else:
        print(f"Frame {f_idx:3d}: (none)")
