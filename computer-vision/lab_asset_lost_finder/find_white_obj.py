import cv2
import numpy as np

f550 = cv2.imread("scratch_preview/theft_24s/frame_550.jpg")
f620 = cv2.imread("scratch_preview/theft_24s/frame_620.jpg")

# The white case is bright white (pixels > 200, 200, 200)
# Region: x: 1650~1850, y: 800~1000
crop550 = f550[800:1000, 1650:1850]
crop620 = f620[800:1000, 1650:1850]

cv2.imwrite("scratch_preview/crop550_white.jpg", crop550)
cv2.imwrite("scratch_preview/crop620_white.jpg", crop620)

diff = cv2.absdiff(crop550, crop620)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
cv2.imwrite("scratch_preview/diff_white.jpg", thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h > 80:
        gx, gy = 1650 + x, 800 + y
        print(f"White object bbox in full frame: [{gx}, {gy}, {gx+w}, {gy+h}], w={w}, h={h}")
