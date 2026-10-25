import pickle
from pathlib import Path

cache_path = Path("output_results/v2_detections.pkl")
if cache_path.exists():
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Total cached frames: {len(cache)}")
    # Look at frame 672 or the last available frame
    last_frame = max(cache.keys())
    persons, items = cache[last_frame]
    print(f"\n--- Frame {last_frame} (Empty Classroom) ---")
    print(f"Persons detected: {len(persons)}")
    print(f"Items detected ({len(items)}):")
    for it in items:
        bx = [int(v) for v in it["bbox"]]
        print(f"  [{it['cname']}] bbox={bx}, w={bx[2]-bx[0]}, h={bx[3]-bx[1]}")

    # Look at frame 72 (In class)
    p72, i72 = cache.get(72, ([], []))
    print(f"\n--- Frame 72 (During Class) ---")
    print(f"Persons detected: {len(p72)}")
    print(f"Items detected ({len(i72)})")
else:
    print("Cache not found at", cache_path)
