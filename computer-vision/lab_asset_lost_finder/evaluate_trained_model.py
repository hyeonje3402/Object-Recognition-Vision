"""
[학습 완료 모델 추론 및 시각화 스크립트 - evaluate_trained_model.py]
커스텀 파인튜닝된 YOLO11 가중치(best.pt)를 로드하여
실습실 CCTV 영상 내의 모든 마우스, 공용 기자재, 에어팟, 충전기 어댑터가 정확하게 감지되는지 검증하고 시각화합니다.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "runs" / "lab_custom_small_items" / "weights" / "best.pt"
INPUT_IMG = BASE_DIR / "dataset" / "images" / "val" / "full_clahe.jpg"
if not INPUT_IMG.exists():
    INPUT_IMG = BASE_DIR / "input_images" / "lab_classroom_cctv.jpg"

OUTPUT_DIR = BASE_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = OUTPUT_DIR / "result_after_custom_training.jpg"

CLASS_COLORS = {
    "mouse": (255, 140, 0),        # 주황색 (개인 마우스)
    "public_mouse": (0, 0, 255),    # 빨간색 (교탁 공용 마우스 - 도난 모니터링)
    "keyboard": (0, 215, 255),     # 노란색 (키보드)
    "airpods": (255, 220, 0),      # 청록/시안 (에어팟/무선이어폰 케이스)
    "charger": (255, 0, 220),      # 마젠타/보라 (충전 어댑터 및 케이블)
}

def evaluate():
    if not WEIGHTS_PATH.exists():
        print(f"[오류] 학습된 가중치 파일을 찾을 수 없습니다: {WEIGHTS_PATH}")
        return

    print("=" * 65)
    print(f"[Evaluation] Loading fine-tuned model from: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    # 추론 실행 (소형 객체 검출 최적화)
    results = model.predict(source=str(INPUT_IMG), conf=0.015, imgsz=640, verbose=False)
    result = results[0]

    img = cv2.imread(str(INPUT_IMG))
    h, w, _ = img.shape

    # UI 오버레이 상단 헤더
    header = np.zeros((75, w, 3), dtype=np.uint8)
    header[:] = (22, 22, 26)
    cv2.putText(header, "[CUSTOM TRAINED YOLO11] Lab Asset & Small Lost Items Detector", 
                (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 255, 200), 2)
    cv2.putText(header, "Classes: Mouse | Public Mouse | Keyboard | AirPods (Earbuds) | Charger Adapter", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

    combined = np.vstack([header, img])
    y_offset = 75

    # ROI 1 영역 표시 (교탁 공용 기자재 영역)
    roi_poly = np.array([[220, 610], [420, 610], [400, 770], [180, 770]], np.int32)
    roi_poly_shifted = roi_poly.copy()
    roi_poly_shifted[:, 1] += y_offset
    cv2.polylines(combined, [roi_poly_shifted], True, (0, 165, 255), 2)
    cv2.putText(combined, "ROI 1: Podium Asset Area", (190, 600 + y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2)

    detections = []
    class_names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        y1_s = y1 + y_offset
        y2_s = y2 + y_offset

        # 박스 그리기
        cv2.rectangle(combined, (x1, y1_s), (x2, y2_s), color, 2)

        # 라벨 배경 및 텍스트
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(combined, (x1, max(0, y1_s - 18)), (x1 + tw + 6, y1_s), color, -1)
        cv2.putText(combined, label, (x1 + 3, y1_s - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append((cls_name, conf, (x1, y1, x2, y2)))

    # 감지 결과 요약 통계 패널 그리기 (우측 하단)
    stat_w, stat_h = 360, 200
    stat_x, stat_y = w - stat_w - 20, h + y_offset - stat_h - 20
    stat_bg = combined[stat_y:stat_y+stat_h, stat_x:stat_x+stat_w].copy()
    black_panel = np.zeros_like(stat_bg)
    cv2.addWeighted(stat_bg, 0.25, black_panel, 0.75, 0, combined[stat_y:stat_y+stat_h, stat_x:stat_x+stat_w])
    cv2.rectangle(combined, (stat_x, stat_y), (stat_x + stat_w, stat_y + stat_h), (0, 255, 200), 1)

    cv2.putText(combined, "INFERENCE DETECTION SUMMARY", (stat_x + 12, stat_y + 25), 
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 200), 1)

    counts = {}
    for d in detections:
        counts[d[0]] = counts.get(d[0], 0) + 1

    lines = [
        f"Public Mouse (Podium): {counts.get('public_mouse', 0)} detected",
        f"Student Mice: {counts.get('mouse', 0)} detected",
        f"Keyboards: {counts.get('keyboard', 0)} detected",
        f"AirPods (Earbuds): {counts.get('airpods', 0)} detected",
        f"Charger / Adapters: {counts.get('charger', 0)} detected",
        f"Total Tracked Lab Items: {len(detections)}"
    ]

    for idx, line in enumerate(lines):
        c = (255, 255, 255) if idx < len(lines)-1 else (0, 255, 200)
        cv2.putText(combined, line, (stat_x + 15, stat_y + 52 + idx * 23), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

    cv2.imwrite(str(OUTPUT_IMG), combined)
    print(f"\n[Success] Inference complete! Result saved to: {OUTPUT_IMG}")
    print(f"Total detections: {len(detections)}")
    for k, v in counts.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    evaluate()
