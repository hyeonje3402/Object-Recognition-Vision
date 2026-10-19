"""
[출입문 전신 보행 진입 및 물품 언패킹 AI 추적 시스템 - render_full_entry_detection.py]
실제 사람의 관절 기구학(Articulated Gait Kinematics)과 자연스러운 보행 스트라이드,
CCTV 조명 음영, 바닥 접촉 그림자(Floor Contact Shadow), 3D 원근 투영 스케일링을 완벽하게 적용하여
사진 사각 오려붙이기 느낌이 전혀 없는 자연스러운 학생 입장 및 물품 언패킹 인식 전 과정을 시뮬레이션합니다.
"""
import cv2
import numpy as np
import math
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMG_SRC = BASE_DIR / "input_images" / "lab_classroom_cctv.jpg"
OUTPUT_DIR = BASE_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT = OUTPUT_DIR / "classroom_person_entry_demo.mp4"
ARTIFACT_DIR = Path(r"C:\Users\82107\.gemini\antigravity-ide\brain\aed0ec8a-0be7-4332-8cf8-7b4013c2f669")

# ==============================================================================
# 1. 인체 비율 및 보행 기구학 기반 자연스러운 학생 렌더링 (Gait Kinematics v2)
# ==============================================================================
def render_realistic_student(phase, scale=1.0, facing_left=True):
    cw, ch = int(200 * scale), int(300 * scale)
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)

    # 자연스러운 보행 기구학 파라미터
    hip_amp = 0.28     # 편안한 일상 보행 보폭 (~16도)
    knee_amp = 0.46    # 자연스러운 무릎 굽힘
    arm_amp = 0.28     # 자연스러운 팔 스윙

    hip_x = cw * 0.50
    bob = math.sin(phase * 2) * (2.4 * scale) # 걸을 때 미세한 인체 상하 바운스
    hip_y = (ch * 0.42) + bob

    thigh_len = 50 * scale
    shin_len = 48 * scale

    # 색상 팔레트 (실제 실습실 학생 복장: 회색 후드티, 다크 차콜 팬츠, 블랙 백팩)
    HOODIE_BASE = np.array([125, 125, 130], dtype=float)
    PANTS_BASE  = np.array([50, 42, 38], dtype=float)
    BACKPACK_COL= np.array([26, 26, 30], dtype=float)
    SKIN_COL    = np.array([145, 178, 218], dtype=float)
    HAIR_COL    = np.array([25, 24, 24], dtype=float)
    SHOE_COL    = np.array([220, 220, 225], dtype=float)
    SHOE_SOLE   = np.array([35, 35, 40], dtype=float)

    def draw_limb(img, p1, p2, r1, r2, base_col, shade_factor=1.0):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3: return
        ux, uy = -dy / dist, dx / dist
        pts = np.array([
            [p1[0] + ux * r1, p1[1] + uy * r1],
            [p1[0] - ux * r1, p1[1] - uy * r1],
            [p2[0] - ux * r2, p2[1] - uy * r2],
            [p2[0] + ux * r2, p2[1] + uy * r2],
        ], dtype=np.int32)
        col = np.clip(base_col * shade_factor, 0, 255).astype(int).tolist()
        cv2.fillPoly(img, [pts], (*col, 255), lineType=cv2.LINE_AA)
        cv2.circle(img, (int(p1[0]), int(p1[1])), int(r1), (*col, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(p2[0]), int(p2[1])), int(r2), (*col, 255), -1, lineType=cv2.LINE_AA)
        edge_col = np.clip(base_col * shade_factor * 0.75, 0, 255).astype(int).tolist()
        cv2.polylines(img, [pts], True, (*edge_col, 180), int(max(1, 1*scale)), lineType=cv2.LINE_AA)

    # 1. 먼 쪽 다리 (Far Leg, 뒤쪽에 살짝 가려진 다리)
    p_far = phase + math.pi
    h_ang_f = math.sin(p_far) * hip_amp
    k_ang_f = max(0.0, -math.sin(p_far - 0.2)) * knee_amp
    kx_f = hip_x - math.sin(h_ang_f) * thigh_len
    ky_f = hip_y + math.cos(h_ang_f) * thigh_len
    ax_f = kx_f - math.sin(h_ang_f + k_ang_f) * shin_len
    ay_f = ky_f + math.cos(h_ang_f + k_ang_f) * shin_len
    draw_limb(canvas, (hip_x, hip_y), (kx_f, ky_f), 13*scale, 9.5*scale, PANTS_BASE, 0.76)
    draw_limb(canvas, (kx_f, ky_f), (ax_f, ay_f), 9.5*scale, 7*scale, PANTS_BASE, 0.74)
    cv2.ellipse(canvas, (int(ax_f - 5*scale), int(ay_f + 4*scale)), 
                (int(13*scale), int(6*scale)), 4, 0, 360, (*(SHOE_COL*0.76).astype(int).tolist(), 255), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(ax_f - 5*scale), int(ay_f + 5.5*scale)), 
                (int(13*scale), int(3*scale)), 4, 0, 360, (*(SHOE_SOLE*0.76).astype(int).tolist(), 255), -1, cv2.LINE_AA)

    # 2. 먼 쪽 팔 (Far Arm)
    p_arm_f = phase + math.pi
    arm_ang_f = -math.sin(p_arm_f) * arm_amp
    shoulder_x = hip_x - (4 * scale) # 걷는 방향으로 미세한 상체 기울임
    shoulder_y = hip_y - (50 * scale)
    e_xf = shoulder_x - math.sin(arm_ang_f) * (34 * scale)
    e_yf = shoulder_y + math.cos(arm_ang_f) * (34 * scale)
    w_xf = e_xf - math.sin(arm_ang_f + 0.15) * (30 * scale)
    w_yf = e_yf + math.cos(arm_ang_f + 0.15) * (30 * scale)
    draw_limb(canvas, (shoulder_x, shoulder_y), (e_xf, e_yf), 9*scale, 7*scale, HOODIE_BASE, 0.8)
    draw_limb(canvas, (e_xf, e_yf), (w_xf, w_yf), 7*scale, 5*scale, HOODIE_BASE, 0.78)
    cv2.circle(canvas, (int(w_xf), int(w_yf)), int(4.5*scale), (*(SKIN_COL*0.8).astype(int).tolist(), 255), -1, cv2.LINE_AA)

    # 3. 몸통 (후드티 자연스러운 볼륨감)
    torso_pts = np.array([
        [shoulder_x - 13*scale, shoulder_y],
        [shoulder_x + 15*scale, shoulder_y + 2*scale],
        [hip_x + 13*scale, hip_y],
        [hip_x - 11*scale, hip_y],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [torso_pts], (*HOODIE_BASE.astype(int).tolist(), 255), lineType=cv2.LINE_AA)
    pocket_pts = np.array([
        [hip_x - 10*scale, hip_y - 15*scale],
        [hip_x + 2*scale,  hip_y - 15*scale],
        [hip_x + 3*scale,  hip_y - 2*scale],
        [hip_x - 8*scale,  hip_y - 2*scale],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pocket_pts], (*(HOODIE_BASE*0.92).astype(int).tolist(), 255), lineType=cv2.LINE_AA)

    # 등에 멘 백팩
    bp_pts = np.array([
        [shoulder_x + 8*scale, shoulder_y + 4*scale],
        [shoulder_x + 25*scale, shoulder_y + 11*scale],
        [shoulder_x + 23*scale, shoulder_y + 44*scale],
        [hip_x + 10*scale, hip_y - 2*scale],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [bp_pts], (*BACKPACK_COL.astype(int).tolist(), 255), lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [bp_pts], True, (55, 55, 60, 255), int(max(1, 1.5*scale)), cv2.LINE_AA)

    # 4. 머리 및 단정한 헤어스타일
    neck_x = shoulder_x - 3*scale
    neck_y = shoulder_y - 8*scale
    cv2.circle(canvas, (int(neck_x), int(neck_y)), int(6.5 * scale), (*SKIN_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)

    head_x = neck_x - 5*scale
    head_y = neck_y - 17*scale
    cv2.ellipse(canvas, (int(head_x), int(head_y)), (int(11*scale), int(15*scale)), -6, 0, 360, (*SKIN_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(head_x + 1*scale), int(head_y - 4*scale)), (int(11.5*scale), int(12.5*scale)), -8, 150, 360, (*HAIR_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(head_x + 5*scale), int(head_y)), int(8.5 * scale), (*HAIR_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)

    # 5. 가까운 쪽 다리 (Near Leg)
    p_near = phase
    h_ang_n = math.sin(p_near) * hip_amp
    k_ang_n = max(0.0, -math.sin(p_near - 0.2)) * knee_amp
    kx_n = hip_x - math.sin(h_ang_n) * thigh_len
    ky_n = hip_y + math.cos(h_ang_n) * thigh_len
    ax_n = kx_n - math.sin(h_ang_n + k_ang_n) * shin_len
    ay_n = ky_n + math.cos(h_ang_n + k_ang_n) * shin_len
    draw_limb(canvas, (hip_x, hip_y), (kx_n, ky_n), 13.5*scale, 10*scale, PANTS_BASE, 1.0)
    draw_limb(canvas, (kx_n, ky_n), (ax_n, ay_n), 10*scale, 7.5*scale, PANTS_BASE, 0.95)
    cv2.ellipse(canvas, (int(ax_n - 5*scale), int(ay_n + 4*scale)), 
                (int(13.5*scale), int(6.5*scale)), 4, 0, 360, (*SHOE_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(ax_n - 5*scale), int(ay_n + 5.5*scale)), 
                (int(13.5*scale), int(3.5*scale)), 4, 0, 360, (*SHOE_SOLE.astype(int).tolist(), 255), -1, cv2.LINE_AA)

    # 6. 가까운 쪽 팔 (Near Arm)
    p_arm_n = phase
    arm_ang_n = -math.sin(p_arm_n) * arm_amp
    e_xn = shoulder_x - math.sin(arm_ang_n) * (34 * scale)
    e_yn = shoulder_y + math.cos(arm_ang_n) * (34 * scale)
    w_xn = e_xn - math.sin(arm_ang_n + 0.15) * (30 * scale)
    w_yn = e_yn + math.cos(arm_ang_n + 0.15) * (30 * scale)
    draw_limb(canvas, (shoulder_x, shoulder_y), (e_xn, e_yn), 9.5*scale, 7.5*scale, HOODIE_BASE, 1.0)
    draw_limb(canvas, (e_xn, e_yn), (w_xn, w_yn), 7.5*scale, 5.5*scale, HOODIE_BASE, 0.98)
    cv2.circle(canvas, (int(w_xn), int(w_yn)), int(5*scale), (*SKIN_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)

    foot_ground_y = max(ay_f, ay_n) + 6*scale
    feet_center_x = (ax_f + ax_n) / 2.0
    return canvas, feet_center_x, foot_ground_y

# ==============================================================================
# 2. 바닥 접촉 그림자 및 안티앨리어싱 합성
# ==============================================================================
def blend_student_onto_frame(frame, ground_x, ground_y, phase, scale=1.0):
    sprite, feet_cx, feet_gy = render_realistic_student(phase, scale=scale)
    sw, sh = sprite.shape[1], sprite.shape[0]

    # 바닥 접촉 그림자
    shadow_w = int(45 * scale)
    shadow_h = int(17 * scale)
    shadow_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.ellipse(shadow_mask, (int(ground_x), int(ground_y)), (shadow_w, shadow_h), -10, 0, 360, 255, -1)
    k_w = int(21 * scale) | 1
    k_h = int(13 * scale) | 1
    shadow_mask = cv2.GaussianBlur(shadow_mask, (k_w, k_h), 0)

    shadow_alpha = (shadow_mask.astype(float) / 255.0) * 0.40
    for c in range(3):
        frame[:, :, c] = np.clip(frame[:, :, c] * (1.0 - shadow_alpha), 0, 255).astype(np.uint8)

    top_x = int(ground_x - feet_cx)
    top_y = int(ground_y - feet_gy)

    x1, y1 = max(0, top_x), max(0, top_y)
    x2, y2 = min(frame.shape[1], top_x + sw), min(frame.shape[0], top_y + sh)

    sx1, sy1 = x1 - top_x, y1 - top_y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    if x2 > x1 and y2 > y1:
        crop_sprite = sprite[sy1:sy2, sx1:sx2]
        spr_alpha = crop_sprite[:, :, 3].astype(float) / 255.0
        spr_rgb = crop_sprite[:, :, :3].astype(float)

        for c in range(3):
            frame[y1:y2, x1:x2, c] = np.clip(
                frame[y1:y2, x1:x2, c] * (1.0 - spr_alpha) + spr_rgb[:, :, c] * spr_alpha, 0, 255
            ).astype(np.uint8)

    # 투명 영역을 제외한 정확한 인체 바운딩 박스 추출
    alpha_pts = np.argwhere(sprite[:, :, 3] > 30)
    if len(alpha_pts) > 0:
        min_sy, min_sx = alpha_pts.min(axis=0)
        max_sy, max_sx = alpha_pts.max(axis=0)
        tight_box = (
            max(0, top_x + min_sx - 4),
            max(0, top_y + min_sy - 4),
            min(frame.shape[1], top_x + max_sx + 4),
            min(frame.shape[0], top_y + max_sy + 4)
        )
    else:
        tight_box = (x1, y1, x2, y2)

    return tight_box

# ==============================================================================
# 3. 전 과정 시뮬레이션 파이프라인
# ==============================================================================
def generate_entry_simulation():
    print("[Simulation] Creating natural articulated walking student entry pipeline...")
    base_img = cv2.imread(str(IMG_SRC))
    if base_img is None:
        print("[Error] 원본 이미지를 찾을 수 없습니다.")
        return

    h, w, _ = base_img.shape
    fps = 10
    total_seconds = 12
    total_frames = fps * total_seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (w, h + 80))

    # 주요 랜드마크
    DOOR_ROI = (1030, 270, 1150, 620)
    DESK_MOUSE = (775, 535, 808, 568)

    # 출입문 문턱부터 앞줄 학생 101 자리까지 실제 타일 바닥을 밟는 보행 경로
    path_points = [
        (1030, 520), # 1. 출입문 문턱 (Door Threshold)
        (1005, 555), # 2. 복도 첫 걸음
        (960, 605),  # 3. 중앙 복도 통과
        (910, 660),  # 4. 앞줄 통로 진입
        (880, 625),  # 5. 책상 방향 전환
        (870, 560)   # 6. 책상 도착 및 착석
    ]

    keyframes = {}
    motion_history = []

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = base_img.copy()

        # ------------------------------------------------------------------
        # 1단계 (0.0s ~ 3.0s): 빈 출입문 상태 및 상시 감시 대기
        # ------------------------------------------------------------------
        if t < 3.0:
            phase_name = "PHASE 1: MONITORING STANDBY & DOORWAY ARMED"
            sub_text = "Status: Doorway Clear | CCTV CH 04 | ROI 2: ARMED (Green) | Ready for Entrance Detection"
            hud_color = (0, 255, 128)

            # 출입문 감시 ROI (초록색)
            cv2.rectangle(frame, (DOOR_ROI[0], DOOR_ROI[1]), (DOOR_ROI[2], DOOR_ROI[3]), (0, 255, 0), 2)
            cv2.putText(frame, "ROI 2: ENTRY DOOR [CLEAR]", (DOOR_ROI[0] - 80, DOOR_ROI[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # 교탁 공용기자재 감시 상태
            cv2.rectangle(frame, (318, 720), (352, 755), (0, 0, 255), 2)
            cv2.putText(frame, "Public Mouse [SAFE]", (280, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

            if frame_idx == 15:
                keyframes["1_empty_lab"] = frame.copy()

        # ------------------------------------------------------------------
        # 2단계 (3.0s ~ 5.5s): 실제 학생이 문을 열고 자연스럽게 첫 걸음을 딛는 순간
        # ------------------------------------------------------------------
        elif 3.0 <= t < 5.5:
            phase_name = "PHASE 2: NATURAL WALKING ENTRY DETECTED"
            sub_text = ">> EVENT: Student Entering Doorway | Natural Gait Motion | Track ID #101 (Conf: 0.95) <<"
            hud_color = (0, 165, 255)

            # 출입문 영역 이벤트 경보 (빨간색)
            cv2.rectangle(frame, (DOOR_ROI[0], DOOR_ROI[1]), (DOOR_ROI[2], DOOR_ROI[3]), (0, 0, 255), 2)
            cv2.putText(frame, "ROI 2: DOOR ENTRY EVENT TRIGGERED!", (DOOR_ROI[0] - 110, DOOR_ROI[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

            # 문턱에서 복도로 자연스럽게 발을 내딛는 보행
            entry_progress = (t - 3.0) / 2.5
            p_gx = int(1030 * (1 - entry_progress) + 1005 * entry_progress)
            p_gy = int(520 * (1 - entry_progress) + 555 * entry_progress)
            
            walk_phase = (t - 3.0) * 6.2
            current_scale = 0.82 + 0.05 * entry_progress

            bbox = blend_student_onto_frame(frame, p_gx, p_gy, phase=walk_phase, scale=current_scale)

            # 학생 바운딩 박스
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 120, 0), 2)
            cv2.putText(frame, "Person #101 [ENTRY] (0.95)", (bbox[0] - 25, bbox[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 120, 0), 2)

            if frame_idx == 40:
                keyframes["2_person_entry"] = frame.copy()

        # ------------------------------------------------------------------
        # 3단계 (5.5s ~ 8.5s): 복도를 따라 다리를 교차하며 걸어오는 전신 추적
        # ------------------------------------------------------------------
        elif 5.5 <= t < 8.5:
            phase_name = "PHASE 3: AISLE TRAJECTORY & ARTICULATED GAIT TRACKING"
            sub_text = "Tracking ID #101: Walking Down Corridor -> Approaching Workstation | Natural Gait Active"
            hud_color = (255, 200, 0)

            progress = (t - 5.5) / 3.0
            idx = min(int(progress * (len(path_points) - 1)), len(path_points) - 2)
            local_alpha = (progress * (len(path_points) - 1)) - idx
            cx = int(path_points[idx][0] * (1 - local_alpha) + path_points[idx+1][0] * local_alpha)
            cy = int(path_points[idx][1] * (1 - local_alpha) + path_points[idx+1][1] * local_alpha)

            motion_history.append((cx, cy))

            # 바닥 이동 궤적 선 (Cyan glowing trajectory)
            for i in range(len(motion_history) - 1):
                cv2.line(frame, motion_history[i], motion_history[i+1], (0, 240, 255), 2, cv2.LINE_AA)
            for pt in motion_history[::4]:
                cv2.circle(frame, pt, 3, (0, 180, 255), -1)

            # 자연스러운 보행 위상 및 원근 스케일링 (0.87 -> 1.05)
            walk_phase = (t - 5.5) * 6.5
            current_scale = 0.87 + 0.18 * progress

            bbox = blend_student_onto_frame(frame, cx, cy, phase=walk_phase, scale=current_scale)

            # 이동 중인 사람 바운딩 박스
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 120, 0), 2)
            cv2.putText(frame, "Person #101 [WALKING GAIT] (0.96)", (bbox[0] - 25, bbox[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 120, 0), 2)

            if frame_idx == 70:
                keyframes["3_moving_to_desk"] = frame.copy()

        # ------------------------------------------------------------------
        # 4단계 (8.5s ~ 12.0s): 책상 착석, 가방에서 마우스 꺼냄 & 실시간 AI 인식 연동
        # ------------------------------------------------------------------
        else:
            phase_name = "PHASE 4: WORKSTATION SEATING & PERSONAL MOUSE ACQUIRED"
            sub_text = "Action: Seated -> Unpacked Mouse from Bag | Mouse Detected (0.97) | Ownership Registered!"
            hud_color = (0, 255, 200)

            # 전체 보행 궤적 표시
            for i in range(len(path_points) - 1):
                cv2.line(frame, path_points[i], path_points[i+1], (0, 240, 255), 2, cv2.LINE_AA)

            # 착석한 학생 상체 바운딩 박스 (실제 학생 #101 위치)
            cv2.rectangle(frame, (810, 480), (925, 650), (255, 140, 0), 2)
            cv2.putText(frame, "Student #101 [SEATED & ACTIVE]", (780, 472), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 140, 0), 2)

            # 바닥에 내려놓은 백팩 감지 박스 (실제 바닥 위 백팩)
            cv2.rectangle(frame, (855, 715), (945, 865), (0, 165, 255), 2)
            cv2.putText(frame, "Backpack [Registered]", (825, 708), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)

            # 책상 위 꺼내진 마우스 실시간 정밀 인식 (책상 상판 위!)
            desk_m = (835, 560, 875, 605)
            cv2.rectangle(frame, (desk_m[0], desk_m[1]), (desk_m[2], desk_m[3]), (0, 255, 128), 2)
            cv2.putText(frame, "Personal Mouse (0.97) [ACQUIRED]", (desk_m[0] - 70, desk_m[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 128), 2)

            # 소유권 연결 벡터선 (Student #101 <=> Personal Mouse)
            cv2.line(frame, (850, 540), (desk_m[0] + 20, desk_m[1]), (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "Binding Vector", (800, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)

            if frame_idx == 105:
                keyframes["4_item_acquired"] = frame.copy()

        # ------------------------------------------------------------------
        # 상단 실시간 모니터링 HUD 대시보드
        # ------------------------------------------------------------------
        hud = np.zeros((80, w, 3), dtype=np.uint8)
        hud[:] = (20, 20, 26)
        cv2.putText(hud, f"[{phase_name}]", (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.72, hud_color, 2)
        cv2.putText(hud, sub_text, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 215, 215), 1)

        timer_str = f"TIME 00:{int(t):02d}s | CCTV CH 04"
        cv2.putText(hud, timer_str, (w - 240, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)

        combined = np.vstack([hud, frame])
        out.write(combined)

    out.release()
    print(f"[성공] 전신 보행 모션 시뮬레이션 비디오 저장 완료: {VIDEO_OUT}")

    # 주요 4단계 스틸컷 이미지 저장 및 아티팩트 복사
    for k_name, k_img in keyframes.items():
        k_path = OUTPUT_DIR / f"flow_{k_name}.jpg"
        cv2.imwrite(str(k_path), k_img)
        print(f" - 스틸컷 저장: {k_path}")
        art_path = ARTIFACT_DIR / f"flow_{k_name}.jpg"
        shutil.copy(str(k_path), str(art_path))

    art_video = ARTIFACT_DIR / "classroom_person_entry_demo.mp4"
    shutil.copy(str(VIDEO_OUT), str(art_video))
    print("[성공] 아티팩트 디렉토리에 최신 결과 복사 완료!")

if __name__ == "__main__":
    generate_entry_simulation()

