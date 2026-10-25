import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("input_videos/merged_cctv_lost_item_video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 670)
ret, frame = cap.read()
cap.release()

model = YOLO("yolov8x.pt")
# Crop left desk area: x: 800~1050, y: 200~450
crop = frame[200:450, 800:1050]
res = model.predict(crop, conf=0.08, verbose=False)[0]

print("Detections on left desk crop at frame 670 (conf >= 0.08):")
for b in res.boxes:
    c = model.names[int(b.cls[0])]
    cf = float(b.conf[0])
    bx = [int(v) for v in b.xyxy[0].tolist()]
    print(f"  {c} ({cf:.2f}): {bx}")
