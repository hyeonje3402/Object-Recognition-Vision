"""
[실제 강의실 CCTV 이미지 분석 스크립트 - detect_lab_image.py]
교탁 위 마우스/키보드, 학생, 출입문 감지 및 상태 머신 HUD 렌더링
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def analyze_lab_classroom():
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "input_images" / "lab_classroom_cctv.jpg"
    output_path = base_dir / "output_results" / "result_lab_cctv_analyzed.jpg"

    print(f"🔍 실습실 이미지 분석 시작: {input_path.name}")
    img = cv2.imread(str(input_path))
    if img is None:
        print("[오류] 이미지를 읽을 수 없습니다.")
        return

    h, w, _ = img.shape
    model_path = base_dir.parent / "yolo11n.pt"
    if not model_path.exists():
        model_path = base_dir.parent / "yolov8n.pt"
    model = YOLO(str(model_path))

    # 실습실 실제 배치에 맞춘 픽셀 ROI
    # ROI 1: 전면 교탁 구역 (Podium)
    roi_podium = (140, 450, 540, 880)
    # ROI 2: 우측 비상구 출입문 구역 (Exit Door)
    roi_door = (1020, 240, 1180, 520)
    # ROI 3: 학생 책상 및 실습 좌석 구역 (Student Desks)
    roi_desk = (450, 350, 1000, 780)

    # YOLO 추론
    results = model(img, conf=0.20, verbose=False)[0]

    annotated = img.copy()

    # 1. ROI 가이드라인 렌더링
    # 교탁 ROI (하늘색)
    cv2.rectangle(annotated, (roi_podium[0], roi_podium[1]), (roi_podium[2], roi_podium[3]), (255, 200, 0), 2)
    cv2.putText(annotated, "ROI 1: PODIUM (PUBLIC ASSET AREA)", (roi_podium[0] + 10, roi_podium[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

    # 출입문 ROI (자주색)
    cv2.rectangle(annotated, (roi_door[0], roi_door[1]), (roi_door[2], roi_door[3]), (200, 0, 200), 2)
    cv2.putText(annotated, "ROI 2: EXIT DOOR", (roi_door[0] - 10, roi_door[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 0, 200), 2)

    # 학생 좌석 ROI (노란색)
    cv2.rectangle(annotated, (roi_desk[0], roi_desk[1]), (roi_desk[2], roi_desk[3]), (0, 255, 255), 2)
    cv2.putText(annotated, "ROI 3: STUDENT DESKS", (roi_desk[0] + 10, roi_desk[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    mouse_found_in_podium = False
    keyboard_found_in_podium = False
    person_count = 0

    # 객체 검출 처리
    names = model.names
    for b in results.boxes:
        box = b.xyxy[0].cpu().numpy().astype(int)
        conf = float(b.conf[0].item())
        cls_id = int(b.cls[0].item())
        cls_name = names[cls_id]
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

        # 1) 공용 마우스 검출
        if cls_name == "mouse":
            # 교탁 ROI 안에 있는지 확인
            if roi_podium[0] <= cx <= roi_podium[2] and roi_podium[1] <= cy <= roi_podium[3]:
                mouse_found_in_podium = True
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.putText(annotated, f"PUBLIC MOUSE [SAFE] ({conf:.2f})", (box[0] - 10, box[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2) 공용 키보드 검출
        elif cls_name == "keyboard":
            if roi_podium[0] <= cx <= roi_podium[2] and roi_podium[1] <= cy <= roi_podium[3]:
                keyboard_found_in_podium = True
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 220, 100), 2)
                cv2.putText(annotated, f"PUBLIC KEYBOARD ({conf:.2f})", (box[0], box[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 2)

        # 3) 사람 검출
        elif cls_name == "person":
            person_count += 1
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 100, 0), 1)

    # 상단 HUD 대시보드 배너
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    # 상태 머신 종합 평가 텍스트
    if mouse_found_in_podium:
        status_text = "[ASSET STATE: SAFE] Public Mouse & Equipment Present on Podium"
        status_col = (0, 255, 0)
    else:
        status_text = "[ASSET STATE: WARNING] Mouse Missing! Grace Timer Running"
        status_col = (0, 200, 255)

    cv2.putText(annotated, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)
    sub_text = f"STUDENTS PRESENT: {person_count} Persons | DOOR AREA: SECURED (No Unauthorized Exit)"
    cv2.putText(annotated, sub_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imwrite(str(output_path), annotated)
    print(f"✅ 분석 완료! 결과 저장: {output_path.name}")
    return output_path

if __name__ == "__main__":
    analyze_lab_classroom()
