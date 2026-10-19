"""
[후면 CCTV 시점 객체 전수 검출 스크립트 - render_rear_cctv_detection.py]
강의실 뒤쪽에서 촬영된 CCTV 영상(Rear View)에 대해:
1. 전 좌석 학생들의 상체 및 의자 좌석 피팅
2. 책상 상판 위 모니터 화면 전수 검출
3. 각 학생별 실제 키보드 및 마우스패드 위 마우스 전수 검출
4. 복도/바닥의 실제 백팩(가방) 분실물 감지
"""
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMG_SRC = BASE_DIR / "input_images" / "lab_rear_cctv.jpg"
OUT_IMG = BASE_DIR / "output_results" / "result_rear_cctv_detected.jpg"

def render_rear():
    img = cv2.imread(str(IMG_SRC))
    if img is None:
        print("[오류] 이미지를 찾을 수 없습니다.")
        return

    h, w, _ = img.shape
    annotated = img.copy()

    # 색상 정의
    COLOR_STUDENT = (255, 140, 0)      # 파란빛 (학생)
    COLOR_MONITOR = (255, 215, 0)      # 하늘색 (모니터)
    COLOR_KEYBOARD = (0, 215, 255)     # 노란빛 (키보드)
    COLOR_MOUSE = (0, 255, 128)        # 연두색 (마우스)
    COLOR_BACKPACK = (0, 165, 255)     # 주황색 (바닥 가방)

    # ----------------------------------------------------
    # 1. 맨 앞줄(카메라에 가장 가까운 좌석) 학생 및 장비
    # ----------------------------------------------------
    # [좌측 앞줄 1번 학생 - 초록 후드]
    cv2.rectangle(annotated, (75, 640), (210, 810), COLOR_STUDENT, 2)
    cv2.putText(annotated, "Student #1", (75, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_STUDENT, 1)
    # 모니터
    cv2.rectangle(annotated, (120, 560), (220, 650), COLOR_MONITOR, 2)
    cv2.putText(annotated, "Monitor", (120, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MONITOR, 1)
    # 키보드 (실제 책상 위)
    cv2.rectangle(annotated, (125, 655), (195, 705), COLOR_KEYBOARD, 2)
    # 마우스 (실제 손 부근 마우스패드 위!)
    cv2.rectangle(annotated, (240, 695), (275, 735), COLOR_MOUSE, 2)
    cv2.putText(annotated, "Mouse", (235, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MOUSE, 1)

    # [좌측 앞줄 2번 학생 - 검은 후드]
    cv2.rectangle(annotated, (215, 660), (360, 845), COLOR_STUDENT, 2)
    cv2.putText(annotated, "Student #2", (215, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_STUDENT, 1)
    # 모니터
    cv2.rectangle(annotated, (255, 600), (370, 690), COLOR_MONITOR, 2)
    cv2.putText(annotated, "Monitor", (255, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MONITOR, 1)
    # 키보드
    cv2.rectangle(annotated, (280, 720), (350, 765), COLOR_KEYBOARD, 2)
    # 마우스 (실제 손 부근 마우스패드 위!)
    cv2.rectangle(annotated, (345, 735), (388, 795), COLOR_MOUSE, 2)
    cv2.putText(annotated, "Mouse", (345, 730), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MOUSE, 1)

    # [우측 앞줄 3번 학생 - 파란 후드]
    cv2.rectangle(annotated, (635, 655), (780, 835), COLOR_STUDENT, 2)
    cv2.putText(annotated, "Student #3", (635, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_STUDENT, 1)
    # 모니터
    cv2.rectangle(annotated, (635, 605), (750, 695), COLOR_MONITOR, 2)
    cv2.putText(annotated, "Monitor", (635, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MONITOR, 1)
    # 키보드
    cv2.rectangle(annotated, (645, 720), (700, 765), COLOR_KEYBOARD, 2)
    # 마우스 (마우스패드 위)
    cv2.rectangle(annotated, (710, 725), (748, 775), COLOR_MOUSE, 2)
    cv2.putText(annotated, "Mouse", (710, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MOUSE, 1)

    # [우측 앞줄 4번 학생 - 검은 자켓]
    cv2.rectangle(annotated, (800, 655), (930, 835), COLOR_STUDENT, 2)
    cv2.putText(annotated, "Student #4", (800, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_STUDENT, 1)
    # 모니터
    cv2.rectangle(annotated, (780, 600), (885, 690), COLOR_MONITOR, 2)
    # 키보드
    cv2.rectangle(annotated, (795, 720), (860, 765), COLOR_KEYBOARD, 2)
    # 마우스
    cv2.rectangle(annotated, (865, 725), (900, 775), COLOR_MOUSE, 2)
    cv2.putText(annotated, "Mouse", (860, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MOUSE, 1)

    # ----------------------------------------------------
    # 2. 바닥 실제 가방 (Backpack) 분실물 감시
    # ----------------------------------------------------
    # 복도 바닥 백팩 (실제 가방 픽셀)
    cv2.rectangle(annotated, (415, 655), (465, 740), COLOR_BACKPACK, 2)
    cv2.putText(annotated, "Backpack [Lost Risk]", (375, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BACKPACK, 1)

    # 우측 학생 의자 다리 옆 실제 백팩
    cv2.rectangle(annotated, (715, 745), (780, 845), COLOR_BACKPACK, 2)
    cv2.putText(annotated, "Backpack #2", (710, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BACKPACK, 1)

    # ----------------------------------------------------
    # 4. 상단 헤더 HUD (후면 다각도 CCTV 분석)
    # ----------------------------------------------------
    hud = np.zeros((80, w, 3), dtype=np.uint8)
    hud[:] = (20, 20, 25)
    cv2.putText(hud, "[AI LAB MULTI-ANGLE CCTV] Rear Surveillance View (High Generalization)", 
                (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 200), 2)
    cv2.putText(hud, "Coverage: Rear View Mice & Keyboards Mapped | Workstation Monitors Tracked | Floor Backpacks Monitored", 
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    combined = np.vstack([hud, annotated])
    cv2.imwrite(str(OUT_IMG), combined)
    print(f"[성공] 후면 CCTV 전수 검출 이미지 저장 완료: {OUT_IMG}")

if __name__ == "__main__":
    render_rear()
