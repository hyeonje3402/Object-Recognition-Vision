import cv2
import numpy as np

# Load full video
cap = cv2.VideoCapture("input_videos/merged_cctv_lost_item_video.mp4")
frame_72, frame_672 = None, None
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    if idx == 72:
        frame_72 = frame.copy()
    elif idx == 672:
        frame_672 = frame.copy()
        break
cap.release()

# Let's check a student laptop box from frame 72, e.g. on row 1 desk or row 2 desk
# In frame 72, student #20 laptop around x: 1950~2150, y: 550~700 (foreground desk student)
# Let's crop from frame_72 and test match in frame_672
crop = frame_72[600:680, 1980:2100]
crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

search_region = frame_672[580:700, 1960:2120]
search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

res = cv2.matchTemplate(search_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
score = res.max()
print(f"Template match score on EMPTY desk where laptop was: {score:.3f}")
