"""
[학생 입장 및 물품 언패킹 인식 시뮬레이션 영상 생성기 - generate_entry_unpack_simulation.py]
시나리오:
1. 처음엔 빈 강의실 (교탁 공용 기자재만 대기 중)
2. 학생 입장 및 좌석 착석 (Student Entry)
3. 가방에서 물품을 꺼내는 행동 감지 (Unpacking Action)
4. 책상 위 물품(마우스, 볼펜, 노트북) 실시간 인식 및 소유자 바인딩 (Item Detection & Owner Binding)
"""
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMG_SRC = BASE_DIR / "input_images" / "lab_classroom_cctv.jpg"
OUTPUT_DIR = BASE_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT = OUTPUT_DIR / "student_entry_unpack_demo.mp4"

def create_simulation_video():
    print("[Simulation] Generating Classroom Entry & Item Unpacking Video...")
    bg_img = cv2.imread(str(IMG_SRC))
    if bg_img is None:
        print("[오류] 원본 배경 이미지를 찾을 수 없습니다.")
        return

    h, w, _ = bg_img.shape
    fps = 10
    total_seconds = 10
    total_frames = fps * total_seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (w, h + 80))

    # 주요 좌표들
    # 교탁 공용 마우스
    PODIUM_MOUSE = (318, 720, 352, 755)
    PODIUM_KEYBOARD = (238, 660, 338, 715)
    
    # 1번 학생 (앞줄 좌측 남학생) 좌석 좌표
    STUDENT_HEAD = (425, 395, 525, 520)
    DESK_SURFACE = (430, 480, 560, 560)
    UNPACKED_MOUSE = (465, 485, 498, 518)
    UNPACKED_PEN = (505, 495, 530, 508)
    UNPACKED_LAPTOP = (440, 510, 540, 565)
    FLOOR_BACKPACK = (395, 520, 440, 600)

    keyframes = {}

    for frame_idx in range(total_frames):
        t = frame_idx / fps # 현재 초 (0.0 ~ 10.0초)
        frame = bg_img.copy()

        # ----------------------------------------------------
        # 단계별 로직 및 시각화
        # ----------------------------------------------------
        if t < 2.5:
            # 1단계: 빈 강의실 (Empty Classroom)
            phase_num = 1
            phase_title = "PHASE 1: EMPTY COMPUTER LAB (SECURE STANDBY)"
            sub_title = "Status: No Students Present | Public Assets Monitored | Personal Items: 0"
            status_color = (0, 255, 128)

            # 학생 영역을 살짝 디밍/빈 자리 처리하여 빈 강의실 연출
            cv2.putText(frame, "[VACANT SEAT]", (430, 460), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)
            cv2.rectangle(frame, (410, 390, 540, 560), (80, 80, 80), 1, cv2.LINE_AA)

            # 교탁 공용 마우스/키보드 상시 감시 표시
            cv2.rectangle(frame, (PODIUM_MOUSE[0], PODIUM_MOUSE[1]), (PODIUM_MOUSE[2], PODIUM_MOUSE[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Public Mouse [LOCKED]", (280, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            if frame_idx == 10:
                keyframes["step1_empty"] = frame.copy()

        elif 2.5 <= t < 5.0:
            # 2단계: 학생 입장 및 착석 (Student Enters & Sits Down)
            phase_num = 2
            phase_title = "PHASE 2: OCCUPANT ENTRY & SEATING DETECTED"
            sub_title = "AI Detection: Student #1 Entered Room -> Occupied Desk #01 | Personal Items: None"
            status_color = (255, 200, 0)

            # 학생 감지 바운딩 박스 (깜빡임 효과 포함)
            cv2.rectangle(frame, (STUDENT_HEAD[0], STUDENT_HEAD[1]), (STUDENT_HEAD[2], STUDENT_HEAD[3]), (255, 140, 0), 2)
            cv2.putText(frame, "Student #1 [ENTERING]", (STUDENT_HEAD[0], STUDENT_HEAD[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)

            # 의자 옆에 멘 가방 감지
            cv2.rectangle(frame, (FLOOR_BACKPACK[0], FLOOR_BACKPACK[1]), (FLOOR_BACKPACK[2], FLOOR_BACKPACK[3]), (0, 165, 255), 2)
            cv2.putText(frame, "Backpack [With Student]", (FLOOR_BACKPACK[0] - 20, FLOOR_BACKPACK[1] - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            if frame_idx == 35:
                keyframes["step2_entry"] = frame.copy()

        elif 5.0 <= t < 7.5:
            # 3단계: 가방에서 물건 꺼내는 행동 감지 (Unpacking Motion)
            phase_num = 3
            phase_title = "PHASE 3: UNPACKING BEHAVIOR DETECTED (ITEM EXTRACTION)"
            sub_title = "Action Recognition: Hand reaching into backpack -> Placing objects on desk"
            status_color = (0, 165, 255)

            # 학생 박스
            cv2.rectangle(frame, (STUDENT_HEAD[0], STUDENT_HEAD[1]), (STUDENT_HEAD[2], STUDENT_HEAD[3]), (255, 140, 0), 2)
            cv2.putText(frame, "Student #1 [ACTIVE]", (STUDENT_HEAD[0], STUDENT_HEAD[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)

            # 손 모션 및 언패킹 ROI 하이라이트
            cv2.rectangle(frame, (DESK_SURFACE[0], DESK_SURFACE[1]), (DESK_SURFACE[2], DESK_SURFACE[3]), (0, 255, 255), 2)
            cv2.putText(frame, ">> UNPACKING MOTION DETECTED <<", (DESK_SURFACE[0] - 30, DESK_SURFACE[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)

            # 물건이 책상에 하나씩 나타나는 트랜지션
            cv2.rectangle(frame, (UNPACKED_MOUSE[0], UNPACKED_MOUSE[1]), (UNPACKED_MOUSE[2], UNPACKED_MOUSE[3]), (0, 255, 255), 1)
            cv2.putText(frame, "Appearing...", (UNPACKED_MOUSE[0] - 10, UNPACKED_MOUSE[1] - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            if frame_idx == 60:
                keyframes["step3_unpacking"] = frame.copy()

        else:
            # 4단계: 물건 실시간 인식 및 소유자 바인딩 확정 (Item Acquired & Bound)
            phase_num = 4
            phase_title = "PHASE 4: REAL-TIME ITEM RECOGNITION & OWNER BINDING"
            sub_title = "Acquired: Mouse (96%), Pen (91%) | Linked to Student #1 | State: SAFE_ACTIVE"
            status_color = (0, 255, 200)

            # 학생 박스
            cv2.rectangle(frame, (STUDENT_HEAD[0], STUDENT_HEAD[1]), (STUDENT_HEAD[2], STUDENT_HEAD[3]), (255, 140, 0), 2)
            cv2.putText(frame, "Student #1 [OWNER LINKED]", (STUDENT_HEAD[0] - 10, STUDENT_HEAD[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)

            # 꺼내진 마우스 정밀 인식
            cv2.rectangle(frame, (UNPACKED_MOUSE[0], UNPACKED_MOUSE[1]), (UNPACKED_MOUSE[2], UNPACKED_MOUSE[3]), (0, 255, 128), 2)
            cv2.putText(frame, "Mouse (0.96) [NEW]", (UNPACKED_MOUSE[0] - 15, UNPACKED_MOUSE[1] - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 2)

            # 꺼내진 볼펜 정밀 인식
            cv2.rectangle(frame, (UNPACKED_PEN[0], UNPACKED_PEN[1]), (UNPACKED_PEN[2], UNPACKED_PEN[3]), (0, 200, 255), 2)
            cv2.putText(frame, "Pen (0.91)", (UNPACKED_PEN[0] - 10, UNPACKED_PEN[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            # 바닥 가방도 안전 등록
            cv2.rectangle(frame, (FLOOR_BACKPACK[0], FLOOR_BACKPACK[1]), (FLOOR_BACKPACK[2], FLOOR_BACKPACK[3]), (0, 165, 255), 2)
            cv2.putText(frame, "Backpack [Registered]", (FLOOR_BACKPACK[0] - 15, FLOOR_BACKPACK[1] - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # 학생과 물품 간 소유권 연결선 (Binding Vector)
            student_center = ((STUDENT_HEAD[0] + STUDENT_HEAD[2]) // 2, STUDENT_HEAD[3])
            mouse_center = ((UNPACKED_MOUSE[0] + UNPACKED_MOUSE[2]) // 2, UNPACKED_MOUSE[1])
            cv2.line(frame, student_center, mouse_center, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "Binding Vector", (student_center[0] - 40, student_center[1] + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)

            if frame_idx == 85:
                keyframes["step4_detected"] = frame.copy()

        # ----------------------------------------------------
        # 상단 실시간 대시보드 HUD 오버레이
        # ----------------------------------------------------
        hud = np.zeros((80, w, 3), dtype=np.uint8)
        hud[:] = (20, 20, 26)
        cv2.putText(hud, f"[{phase_title}]", (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2)
        cv2.putText(hud, sub_title, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)

        # 시간 및 프레임 표시
        time_text = f"REC 00:0{int(t):02d}s | FPS: {fps}"
        cv2.putText(hud, time_text, (w - 220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        combined = np.vstack([hud, frame])
        out.write(combined)

    out.release()
    print(f"[Success] Video saved to: {VIDEO_OUT}")

    # 주요 4단계 스크린샷 이미지 저장
    for key, kframe in keyframes.items():
        kpath = OUTPUT_DIR / f"sim_{key}.jpg"
        cv2.imwrite(str(kpath), kframe)
        print(f" - Saved keyframe: {kpath}")

if __name__ == "__main__":
    create_simulation_video()
