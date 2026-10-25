import cv2
import numpy as np

f550 = cv2.imread("scratch_preview/theft_24s/frame_550.jpg")
f620 = cv2.imread("scratch_preview/theft_24s/frame_620.jpg")

# Teacher desk area: y: 720~1150, x: 1350~2250
desk550 = f550[720:1150, 1350:2250]
desk620 = f620[720:1150, 1350:2250]

diff = cv2.absdiff(desk550, desk620)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

cv2.imwrite("scratch_preview/desk_diff.jpg", thresh)
cv2.imwrite("scratch_preview/desk550_full.jpg", desk550)
cv2.imwrite("scratch_preview/desk620_full.jpg", desk620)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} diff regions on desk:")
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h > 100:
        global_x1, global_y1 = 1350 + x, 720 + y
        print(f"  Diff region at global [{global_x1}, {global_y1}, {global_x1 + w}, {global_y1 + h}], area={w*h}")
