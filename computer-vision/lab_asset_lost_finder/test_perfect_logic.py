import pickle
from pathlib import Path

cache_path = Path("output_results/v2_detections.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

print(f"Loaded cache with {len(cache)} detection frames.")

# Let's inspect all frames to see what items appear and disappear
# 1. Establish initial desk items from frames 1~120 (during class)
item_history = {}
for f_idx in sorted(cache.keys()):
    if f_idx > 120:
        break
    persons, dets = cache[f_idx]
    for d in dets:
        cname = d["cname"]
        bx = d["bbox"]
        # Skip podium (teacher items)
        if bx[1] > 950:
            continue
        # Skip very large boxes
        if (bx[2] - bx[0]) > 250 or (bx[3] - bx[1]) > 220:
            continue
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        # Match with established items
        matched = False
        for k, v in item_history.items():
            if v["cname"] == cname:
                dcx, dcy = (v["bbox"][0] + v["bbox"][2]) / 2, (v["bbox"][1] + v["bbox"][3]) / 2
                if (cx - dcx)**2 + (cy - dcy)**2 < 80**2:
                    v["hits"] += 1
                    v["last_seen"] = f_idx
                    matched = True
                    break
        if not matched:
            item_id = len(item_history) + 1
            item_history[item_id] = {
                "cname": cname, "bbox": bx, "hits": 1, "first_seen": f_idx, "last_seen": f_idx
            }

confirmed_items = {k: v for k, v in item_history.items() if v["hits"] >= 3}
print(f"Confirmed initial items during class: {len(confirmed_items)}")
for k, v in confirmed_items.items():
    bx = [int(x) for x in v["bbox"]]
    print(f"  Item #{k} [{v['cname']}]: bbox={bx}, hits={v['hits']}")

# Now let's check which of these items are still detected in the last 100 frames (empty room: frames 576~672)
print("\nChecking presence in empty room (frames >= 576):")
late_frames = [f for f in sorted(cache.keys()) if f >= 576]
for k, v in confirmed_items.items():
    cname = v["cname"]
    vx1, vy1, vx2, vy2 = v["bbox"]
    vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
    seen_in_late = 0
    for lf in late_frames:
        _, dets = cache[lf]
        for d in dets:
            if d["cname"] == cname:
                dcx, dcy = (d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2
                if (vcx - dcx)**2 + (vcy - dcy)**2 < 100**2:
                    seen_in_late += 1
                    break
    ratio = seen_in_late / len(late_frames)
    status = "STILL PRESENT (LOST)" if ratio > 0.4 else "TAKEN / RETRIEVED"
    print(f"  Item #{k} [{cname}] at ({int(vcx)}, {int(vcy)}): seen in {seen_in_late}/{len(late_frames)} frames -> {status}")
