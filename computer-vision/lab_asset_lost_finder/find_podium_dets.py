import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("input_videos/merged_cctv_lost_item_video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 550)
ret, f550 = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 576)
ret, f576 = cap.read()
cap.release()

model = YOLO("yolov8x.pt")
res550 = model.predict(f550, conf=0.15, verbose=False)[0]
print("Detections on frame 550 near podium (y > 700):")
for b in res550.boxes:
    bx = [int(v) for v in b.xyxy[0].tolist()]
    if bx[1] > 700:
        cname = model.names[int(b.cls[0])]
        cf = float(b.conf[0])
        print(f"  {cname} ({cf:.2f}): {bx}")

res576 = model.predict(f576, conf=0.15, verbose=False)[0]
print("\nDetections on frame 576 near podium (y > 700):")
for b in res576.boxes:
    bx = [int(v) for v in b.xyxy[0].tolist()]
    if bx[1] > 700:
        cname = model.names[int(b.cls[0])]
        cf = float(b.conf[0])
        print(f"  {cname} ({cf:.2f}): {bx}")
