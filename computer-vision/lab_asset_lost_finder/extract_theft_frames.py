import cv2
from pathlib import Path

cap = cv2.VideoCapture("input_videos/merged_cctv_lost_item_video.mp4")
out_dir = Path("scratch_preview/theft_24s")
out_dir.mkdir(parents=True, exist_ok=True)

# 24s at 24fps is frame 576. Let's extract frames 550, 565, 576, 585, 600, 620
frames_to_save = [550, 565, 576, 585, 600, 620]
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    if idx in frames_to_save:
        cv2.imwrite(str(out_dir / f"frame_{idx}.jpg"), frame)
        print(f"Saved frame {idx}")
    if idx > max(frames_to_save):
        break
cap.release()
print("Extracted frames around 24 seconds successfully!")
