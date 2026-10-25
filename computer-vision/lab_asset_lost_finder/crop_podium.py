import cv2

cap = cv2.VideoCapture("input_videos/merged_cctv_lost_item_video.mp4")

# Read frame 550 and frame 620
cap.set(cv2.CAP_PROP_POS_FRAMES, 550)
ret, f550 = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 620)
ret, f620 = cap.read()
cap.release()

# Teacher desk region around the white mouse / remote
# Look at user screenshot: The hand is at [x: 1720~1820, y: 820~900]
crop550 = f550[750:1050, 1400:2000]
crop620 = f620[750:1050, 1400:2000]

cv2.imwrite("scratch_preview/podium_550.jpg", crop550)
cv2.imwrite("scratch_preview/podium_620.jpg", crop620)
print("Saved podium crops at 550 and 620")
