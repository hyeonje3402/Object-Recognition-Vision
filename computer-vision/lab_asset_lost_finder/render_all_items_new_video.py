"""
[신규 CCTV 영상 전수 객체 탐지 및 정밀 시각화 - render_all_items_new_video.py]
2944x1248 초고해상도 고정 CCTV 영상을 YOLOv8x로 정밀 분석하여
실습실 내 모든 학생, 공용 모니터, 노트북, 마우스, 키보드, 스마트폰, 텀블러, 가방 등을 빠짐없이 전수 인식하고 시각화합니다.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "input_videos" / "Hailuo_Video_STRICT SCENE AND CAMERA LOCK_ _570390103425851395.mp4"
FRAME_PATH = BASE_DIR / "scratch_preview" / "new_video_frame_0_0.jpg"
OUTPUT_DIR = BASE_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = OUTPUT_DIR / "new_video_all_items_detected.jpg"

# 물품 카테고리별 고대비 색상 팔레트 (BGR)
COLOR_PALETTE = {
    "person": (255, 150, 0),       # 파란색 (학생)
    "laptop": (255, 80, 180),      # 보라/핑크 (노트북 본체/화면)
    "tv": (255, 220, 50),          # 하늘색 (고정 모니터)
    "mouse": (0, 140, 255),        # 주황색 (마우스 - 핵심 추적 대상)
    "keyboard": (0, 215, 255),     # 노란색 (키보드)
    "cell phone": (255, 0, 255),   # 마젠타 (스마트폰)
    "backpack": (50, 180, 255),    # 코랄 (바닥 백팩)
    "handbag": (50, 180, 255),     # 코랄 (가방)
    "bottle": (100, 255, 120),     # 민트그린 (텀블러/물병)
    "cup": (100, 255, 120),        # 민트그린 (컵)
    "chair": (70, 70, 80),         # 어두운 회색 (의자 - 배경)
    "clock": (200, 200, 200),      # 연회색 (벽시계)
    "remote": (0, 255, 200),       # 민트 (리모컨)
}

def render_all_items():
    print("=" * 65)
    print("🚀 [전수 객체 탐지 시작] 신규 고정 CCTV 영상 정밀 분석")
    print(f" - 입력 영상: {VIDEO_PATH.name}")
    print(f" - 기준 프레임: {FRAME_PATH.name}")
    print(" - 탐지 모델: YOLOv8x (Extra Large High-Precision)")
    print("=" * 65)

    if not FRAME_PATH.exists():
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[오류] 영상에서 프레임을 읽을 수 없습니다.")
            return
        cv2.imwrite(str(FRAME_PATH), frame)

    img = cv2.imread(str(FRAME_PATH))
    h, w, _ = img.shape
    print(f" - 원본 해상도: {w} x {h}")

    # YOLOv8x 고해상도(imgsz=1920) 추론
    model = YOLO(str(BASE_DIR / "yolov8x.pt"))
    results = model.predict(img, imgsz=1920, conf=0.15, verbose=False)[0]

    annotated = img.copy()

    # 상단 관제 HUD 헤더 바 생성
    hud_h = 80
    header = np.zeros((hud_h, w, 3), dtype=np.uint8)
    header[:] = (20, 20, 24)

    cv2.putText(header, "[SMART LAB CCTV] Ultra-Wide High-Resolution Multi-Asset Detection (2944x1248)",
                (25, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 200), 2)
    cv2.putText(header, f"Video: {VIDEO_PATH.name[:55]}... | Model: YOLOv8x High-Precision | Mode: Full Inventory Mapping",
                (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    combined = np.vstack([header, annotated])
    y_shift = hud_h

    counts = {}
    total_objects = 0

    # 1. 감지된 객체 정렬 (크기순: 큰 객체인 사람/의자를 먼저 그리고, 작은 마우스/폰을 위에 렌더링)
    box_list = []
    for b in results.boxes:
        c_name = model.names[int(b.cls[0])]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        box_list.append((area, c_name, conf, x1, y1, x2, y2))

    box_list.sort(key=lambda item: item[0], reverse=True)

    for _, c_name, conf, x1, y1, x2, y2 in box_list:
        total_objects += 1
        counts[c_name] = counts.get(c_name, 0) + 1

        color = COLOR_PALETTE.get(c_name, (200, 200, 200))
        y1_s = y1 + y_shift
        y2_s = y2 + y_shift

        # 의자(chair)는 배경이므로 얇고 은은하게 처리
        if c_name == "chair":
            cv2.rectangle(combined, (x1, y1_s), (x2, y2_s), (60, 60, 70), 1)
            continue

        # 중요 기자재 및 학생 바운딩 박스
        thickness = 2 if c_name in ["mouse", "cell phone", "keyboard", "laptop", "tv"] else 1
        cv2.rectangle(combined, (x1, y1_s), (x2, y2_s), color, thickness)

        # 라벨 배지 렌더링
        label = f"{c_name.upper()} {conf:.2f}"
        font_scale = 0.36 if c_name in ["mouse", "cell phone", "keyboard", "bottle", "cup"] else 0.42
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        badge_y1 = max(y_shift, y1_s - th - 6)
        badge_y2 = y1_s
        cv2.rectangle(combined, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)
        cv2.putText(combined, label, (x1 + 3, badge_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # 우측 상단 실시간 관제 통계 대시보드 패널 오버레이
    stat_w, stat_h = 420, 260
    stat_x = w - stat_w - 25
    stat_y = y_shift + 25

    panel_roi = combined[stat_y:stat_y + stat_h, stat_x:stat_x + stat_w].copy()
    dark_mask = np.zeros_like(panel_roi)
    cv2.addWeighted(panel_roi, 0.25, dark_mask, 0.75, 0, combined[stat_y:stat_y + stat_h, stat_x:stat_x + stat_w])
    cv2.rectangle(combined, (stat_x, stat_y), (stat_x + stat_w, stat_y + stat_h), (0, 255, 200), 1)

    cv2.putText(combined, "LAB CCTV DETECTION SUMMARY", (stat_x + 15, stat_y + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 200), 1)

    summary_lines = [
        f"Total Detected Assets: {total_objects} items",
        f"- Students (Person): {counts.get('person', 0)} seated",
        f"- Laptops (Notebook): {counts.get('laptop', 0)} active",
        f"- Fixed Monitors (TV): {counts.get('tv', 0)} units",
        f"- Mice (Tracked): {counts.get('mouse', 0)} units",
        f"- Keyboards: {counts.get('keyboard', 0)} units",
        f"- Smartphones: {counts.get('cell phone', 0)} detected",
        f"- Backpacks & Bags: {counts.get('backpack', 0) + counts.get('handbag', 0)} on floor",
        f"- Tumblers & Cups: {counts.get('bottle', 0) + counts.get('cup', 0)} on desks",
    ]

    for idx, text in enumerate(summary_lines):
        t_color = (0, 255, 200) if idx == 0 else (240, 240, 240)
        cv2.putText(combined, text, (stat_x + 18, stat_y + 55 + idx * 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, t_color, 1)

    cv2.imwrite(str(OUTPUT_IMG), combined)
    print(f"\n[성공] 전수 객체 탐지 시각화 완료!")
    print(f" - 결과 이미지 저장: {OUTPUT_IMG}")
    print(f" - 총 탐지 객체 수: {total_objects}개")
    for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   * {k}: {v}")

if __name__ == "__main__":
    render_all_items()
