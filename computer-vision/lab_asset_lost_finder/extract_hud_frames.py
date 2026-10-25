import cv2
from pathlib import Path

video_path = Path("output_results/merged_cctv_lost_item_analyzed.mp4")
out_dir = Path("scratch_preview/analyzed_hud_frames")
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
frames_to_save = [80, 360, 500, 650]
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    if idx in frames_to_save:
        out_path = out_dir / f"hud_frame_{idx}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"Saved: {out_path}")
    if idx > max(frames_to_save):
        break
cap.release()
print("Preview extraction complete!")
