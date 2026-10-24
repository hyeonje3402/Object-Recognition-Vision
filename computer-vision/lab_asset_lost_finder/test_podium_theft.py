import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from core.interaction_tracker import DynamicTheftDetector, TheftState, AssetPresenceChecker

VIDEO_PATH = Path("input_videos/merged_cctv_lost_item_video.mp4")
cap = cv2.VideoCapture(str(VIDEO_PATH))

# Registered podium assets
assets = {
    "PODIUM_MOUSE": {"class": "mouse", "bbox": [1768, 1042, 1805, 1078], "conf": 0.85},
    "PODIUM_REMOTE": {"class": "remote", "bbox": [1572, 1019, 1655, 1091], "conf": 0.80},
}

# Baseline frame 500 (before student approaches)
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, f500 = cap.read()
presence_checker = AssetPresenceChecker(f500, assets, search_pad=25, match_thresh=0.50)

# Exit door at top left: [0, 0, 450, 600]
DOOR_ROI = (0, 0, 450, 600)
detector = DynamicTheftDetector(
    exit_door_bbox=DOOR_ROI,
    fps=24.0,
    interact_dist=220.0,
    confirm_missing_sec=0.4,
    interaction_memory_sec=8.0,
    door_trigger_dist=350.0,
    approach_px=100.0,
    min_departure_px=120.0
)

# Test from frame 520 to 650
cap.set(cv2.CAP_PROP_POS_FRAMES, 520)
tracker_model = YOLO("yolo11n.pt")

events = []
frame_idx = 520
while cap.isOpened() and frame_idx < 650:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Track persons with ByteTrack
    res = tracker_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                              conf=0.25, imgsz=1280, verbose=False)[0]
    persons = []
    if res.boxes is not None and res.boxes.id is not None:
        for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()):
            persons.append({"track_id": int(tid), "bbox": [int(v) for v in box]})

    # Check asset presence
    scores = presence_checker.check(frame, assets)
    presence = presence_checker.presence(scores)

    # Update theft detector
    evs = detector.update(frame_idx, assets, presence, persons)
    if evs:
        for e in evs:
            sec = frame_idx / 24.0
            print(f"THEFT DETECTED: Frame {frame_idx} ({sec:.2f}s) - Asset: {e['asset_id']}, Suspect: #{e['suspect_id']}, Door Dist: {e.get('door_distance')}, Departure: {e.get('departure_px')}px")
        events.extend(evs)

cap.release()
print(f"\nTotal theft events detected: {len(events)}")
