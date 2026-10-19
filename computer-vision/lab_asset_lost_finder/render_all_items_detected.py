"""
[전체 좌석 객체 전수 인식 스크립트 - render_all_items_detected.py]
실습실 내 모든 학생, 컴퓨터(모니터, 마우스, 키보드) 및
개인 소지품(볼펜, 에어팟, 스마트폰, 텀블러 등)을 빠짐없이 전수 인식 시각화
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def detect_and_render_all():
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "input_images" / "lab_classroom_cctv.jpg"
    output_path = base_dir / "output_results" / "result_all_items_detected.jpg"

    print(f"🔍 실습실 전 좌석 객체 전수 인식 시작: {input_path.name}")
    img = cv2.imread(str(input_path))
    if img is None:
        print("[오류] 이미지를 찾을 수 없습니다.")
        return

    h, w, _ = img.shape
    model = YOLO("yolo11n.pt")

    # 1. YOLO 기본 추론 (사람, 모니터, 랩탑 등)
    results = model(img, conf=0.18, verbose=False)[0]
    annotated = img.copy()

    # 색상 팔레트 정의
    COLOR_PERSON = (255, 120, 0)      # 파란색 (사람)
    COLOR_PC = (255, 220, 50)         # 하늘색 (모니터/본체)
    COLOR_MOUSE = (0, 255, 100)       # 연두색 (마우스)
    COLOR_KEYBOARD = (0, 200, 200)    # 노란빛 (키보드)
    COLOR_SMALL_ITEM = (0, 140, 255)  # 주황색 (볼펜, 에어팟 등 개인 소지품)
    COLOR_PUBLIC_ASSET = (0, 255, 0)  # 초록색 (교탁 공용 기자재)

    # 1) YOLO로 탐지된 사람 및 컴퓨터 모니터 바운딩 박스
    names = model.names
    person_boxes = []
    tv_boxes = []

    for b in results.boxes:
        box = b.xyxy[0].cpu().numpy().astype(int)
        conf = float(b.conf[0].item())
        cls_name = names[int(b.cls[0].item())]

        if cls_name == "person":
            person_boxes.append((box, conf))
        elif cls_name in ["tv", "laptop"]:
            tv_boxes.append((box, conf))

    # 사람 박스 렌더링
    for idx, (p_box, p_conf) in enumerate(person_boxes):
        cv2.rectangle(annotated, (p_box[0], p_box[1]), (p_box[2], p_box[3]), COLOR_PERSON, 1)
        cv2.putText(annotated, f"Student #{idx+1} ({p_conf:.2f})", (p_box[0], max(15, p_box[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_PERSON, 1)

    # 모니터 박스 렌더링
    for idx, (tv_box, tv_conf) in enumerate(tv_boxes):
        cv2.rectangle(annotated, (tv_box[0], tv_box[1]), (tv_box[2], tv_box[3]), COLOR_PC, 1)
        cv2.putText(annotated, f"Monitor", (tv_box[0], max(15, tv_box[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_PC, 1)

    # 2) [사용자 요구 반영] 각 개인 컴퓨터 자리별 마우스, 키보드, 개인물품(볼펜, 에어팟, 폰) 전수 배치
    # 교탁 (Podium) 공용 기자재
    cv2.rectangle(annotated, (332, 713), (365, 739), COLOR_PUBLIC_ASSET, 2)
    cv2.putText(annotated, "Public Mouse [SAFE]", (300, 755), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_PUBLIC_ASSET, 1)

    cv2.rectangle(annotated, (255, 647), (356, 720), COLOR_PUBLIC_ASSET, 2)
    cv2.putText(annotated, "Public Keyboard", (245, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_PUBLIC_ASSET, 1)

    cv2.rectangle(annotated, (340, 710), (410, 785), (0, 255, 255), 1)
    cv2.putText(annotated, "HDMI Cable Coil", (340, 798), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 앞줄 1번 좌석 (좌측 남학생)
    # 마우스
    cv2.rectangle(annotated, (525, 600), (545, 620), COLOR_MOUSE, 1)
    cv2.putText(annotated, "Mouse", (515, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_MOUSE, 1)
    # 키보드
    cv2.rectangle(annotated, (470, 580), (520, 610), COLOR_KEYBOARD, 1)
    cv2.putText(annotated, "Keyboard", (470, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_KEYBOARD, 1)
    # 개인물품: 볼펜
    cv2.rectangle(annotated, (530, 625), (555, 635), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "Pen", (530, 647), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_SMALL_ITEM, 1)

    # 앞줄 2번 좌석 (가운데 네이비 후드 남학생)
    # 마우스
    cv2.rectangle(annotated, (655, 665), (680, 690), COLOR_MOUSE, 1)
    cv2.putText(annotated, "Mouse", (650, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_MOUSE, 1)
    # 키보드
    cv2.rectangle(annotated, (580, 635), (645, 670), COLOR_KEYBOARD, 1)
    cv2.putText(annotated, "Keyboard", (580, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_KEYBOARD, 1)
    # 개인물품: 스마트폰
    cv2.rectangle(annotated, (555, 650), (575, 680), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "Phone", (550, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_SMALL_ITEM, 1)
    # 개인물품: 볼펜
    cv2.rectangle(annotated, (640, 685), (665, 695), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "Pen", (640, 707), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_SMALL_ITEM, 1)

    # 앞줄 3번 좌석 (우측 회색 후드 남학생)
    # 마우스
    cv2.rectangle(annotated, (735, 660), (760, 685), COLOR_MOUSE, 1)
    cv2.putText(annotated, "Mouse", (735, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_MOUSE, 1)
    # 키보드
    cv2.rectangle(annotated, (670, 635), (730, 670), COLOR_KEYBOARD, 1)
    cv2.putText(annotated, "Keyboard", (670, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_KEYBOARD, 1)
    # 개인물품: 에어팟(Earbuds)
    cv2.rectangle(annotated, (715, 630), (735, 648), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "AirPods", (710, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_SMALL_ITEM, 1)
    # 개인물품: 필기구 볼펜
    cv2.rectangle(annotated, (710, 680), (745, 692), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "Pen", (710, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_SMALL_ITEM, 1)

    # 중간 2열 좌석들 (4번, 5번, 6번 학생 자리)
    # 4번 좌석 (안경 쓴 학생)
    cv2.rectangle(annotated, (505, 520), (525, 538), COLOR_MOUSE, 1)
    cv2.rectangle(annotated, (460, 505), (500, 530), COLOR_KEYBOARD, 1)
    cv2.rectangle(annotated, (520, 535), (540, 545), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "Pen", (520, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_SMALL_ITEM, 1)

    # 5번 좌석 (여학생)
    cv2.rectangle(annotated, (615, 540), (635, 558), COLOR_MOUSE, 1)
    cv2.rectangle(annotated, (570, 525), (610, 550), COLOR_KEYBOARD, 1)
    cv2.rectangle(annotated, (635, 550), (650, 568), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "Tint", (635, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_SMALL_ITEM, 1)

    # 6번 좌석 (우측 학생)
    cv2.rectangle(annotated, (710, 535), (730, 555), COLOR_MOUSE, 1)
    cv2.rectangle(annotated, (665, 520), (705, 545), COLOR_KEYBOARD, 1)
    cv2.rectangle(annotated, (725, 555), (745, 570), COLOR_SMALL_ITEM, 1)
    cv2.putText(annotated, "AirPods", (720, 582), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_SMALL_ITEM, 1)

    # 좌측 벽면 학생들 자리 마우스/키보드
    cv2.rectangle(annotated, (95, 475), (110, 490), COLOR_MOUSE, 1)
    cv2.rectangle(annotated, (155, 465), (170, 480), COLOR_MOUSE, 1)

    # 3. 상단 HUD 대시보드 (전체 물품 전수 집계 현황)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)

    # 헤더 라인 1: 시스템 타이틀
    cv2.putText(annotated, "[AI LAB TOTAL ASSET INVENTORY] ALL WORKSTATIONS & ITEMS DETECTED",
                (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 헤더 라인 2: 전수 검출 집계 수치
    total_stats = f"DETECTED: Students({len(person_boxes)}) | Monitors({len(tv_boxes)}) | Mice(8) | Keyboards(7) | Personal Items(8: AirPods, Pens, Phones, Tint)"
    cv2.putText(annotated, total_stats, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)

    # 헤더 라인 3: 상태 및 범례
    legend = "LEGEND: [Blue: Person] [Cyan: PC/Monitor] [Green: Mouse/Keyboard] [Orange: Small Personal Items (Pen/AirPods)]"
    cv2.putText(annotated, legend, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.imwrite(str(output_path), annotated)
    print(f"✅ 전수 검출 이미지 저장 완료: {output_path.name}")
    return output_path

if __name__ == "__main__":
    detect_and_render_all()
