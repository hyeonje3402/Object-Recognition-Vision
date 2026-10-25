import cv2
from pathlib import Path

video_path = Path("output_results/merged_cctv_lost_item_analyzed_v2.mp4")
out_dir = Path("scratch_preview/v2_frames")
out_dir.mkdir(parents=True, exist_ok=True)

targets = {150, 300, 420, 430, 676}
cap = cv2.VideoCapture(str(video_path))
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    if idx in targets:
        small = cv2.resize(frame, (1472, 624))
        cv2.imwrite(str(out_dir / f"v2_{idx}.jpg"), small)
        print("saved", idx)
    if idx >= max(targets):
        break
cap.release()
