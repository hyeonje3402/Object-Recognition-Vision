import cv2
import numpy as np

f550 = cv2.imread("scratch_preview/theft_24s/frame_550.jpg")
f620 = cv2.imread("scratch_preview/theft_24s/frame_620.jpg")

desk550 = f550[720:1150, 1350:2250]
desk620 = f620[720:1150, 1350:2250]

diff = cv2.absdiff(desk550, desk620)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

# Find contours in the region around the airpods (rx: 400~500, ry: 150~250)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
airpods_box = None
max_area = 0
for c in contours:
    rx, ry, rw, rh = cv2.boundingRect(c)
    if 350 <= rx <= 550 and 150 <= ry <= 300:
        area = rw * rh
        if area > max_area:
            max_area = area
            airpods_box = [1350 + rx, 720 + ry, 1350 + rx + rw, 720 + ry + rh]

print("AirPods bounding box:", airpods_box)
# Let's crop it from frame 550 to visually confirm
bx = airpods_box
pad = 20
crop = f550[max(0, bx[1]-pad):bx[3]+pad, max(0, bx[0]-pad):bx[2]+pad]
cv2.imwrite("scratch_preview/airpods_crop.jpg", crop)
print("Saved airpods_crop.jpg")
