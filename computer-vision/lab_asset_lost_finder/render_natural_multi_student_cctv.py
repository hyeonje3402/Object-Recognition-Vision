"""
[실시간 실습실 전원 자연스러운 모션 & AI 정밀 감시 시스템 - render_natural_multi_student_cctv.py]
- 한 사람만 움직이거나 배경 학생들이 굳어있는 현상 100% 해결
- 실습실 내 모든 학생(1열 3명, 2열 4명, 벽면 2명, 신규 진입 학생)이 각자 자연스러운 신체 모션 수행:
  * 학생 #1: 마우스 조작, 화면 주시 고개 끄덕임, 호흡
  * 학생 #2: 키보드 양손 타이핑, 좌우 시선 전환, 상체 호흡
  * 학생 #3: 시험지/노트 위 볼펜 필기 모션(펜 실시간 연동), 화면/노트 번갈아 응시
  * 2열 학생들: 화면 주시, 키보드 조작, 미세 자세 조정
  * 벽면 학생들: 모니터 작업 및 상체 움직임
  * 진입 학생: 출입문 개방 -> 복도 자연 보행 -> 책상 착석 -> 마우스 언패킹 및 소유권 등록
- 모든 객체(학생, 공용기자재, 개인소지품)의 바운딩 박스가 인체 움직임에 맞춰 실시간 유기적 추적(Dynamic Tracking)
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

VIDEO_OUT_MULTI = OUTPUT_DIR / "classroom_natural_multi_student_cctv.mp4"
VIDEO_OUT_ENTRY = OUTPUT_DIR / "classroom_person_entry_demo.mp4"

CURR_ART_DIR = Path(r"C:\Users\82107\.gemini\antigravity-ide\brain\046a9fed-287d-4467-afe6-a298d3d9e04b")
PREV_ART_DIR = Path(r"C:\Users\82107\.gemini\antigravity-ide\brain\aed0ec8a-0be7-4332-8cf8-7b4013c2f669")
for d in [CURR_ART_DIR, PREV_ART_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. 자연스러운 보행 학생 렌더링 (Gait Kinematics v2)
# ==============================================================================
def render_realistic_student(phase, scale=1.0):
    cw, ch = int(200 * scale), int(300 * scale)
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)

    hip_amp = 0.28
    knee_amp = 0.46
    arm_amp = 0.28

    hip_x = cw * 0.50
    bob = math.sin(phase * 2) * (2.4 * scale)
    hip_y = (ch * 0.42) + bob

    thigh_len = 50 * scale
    shin_len = 48 * scale

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

    # 1. 먼 쪽 다리
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

    # 2. 먼 쪽 팔
    p_arm_f = phase + math.pi
    arm_ang_f = -math.sin(p_arm_f) * arm_amp
    shoulder_x = hip_x - (4 * scale)
    shoulder_y = hip_y - (50 * scale)
    e_xf = shoulder_x - math.sin(arm_ang_f) * (34 * scale)
    e_yf = shoulder_y + math.cos(arm_ang_f) * (34 * scale)
    w_xf = e_xf - math.sin(arm_ang_f + 0.15) * (30 * scale)
    w_yf = e_yf + math.cos(arm_ang_f + 0.15) * (30 * scale)
    draw_limb(canvas, (shoulder_x, shoulder_y), (e_xf, e_yf), 9*scale, 7*scale, HOODIE_BASE, 0.8)
    draw_limb(canvas, (e_xf, e_yf), (w_xf, w_yf), 7*scale, 5*scale, HOODIE_BASE, 0.78)
    cv2.circle(canvas, (int(w_xf), int(w_yf)), int(4.5*scale), (*(SKIN_COL*0.8).astype(int).tolist(), 255), -1, cv2.LINE_AA)

    # 3. 몸통 & 백팩
    torso_pts = np.array([
        [shoulder_x - 13*scale, shoulder_y],
        [shoulder_x + 15*scale, shoulder_y + 2*scale],
        [hip_x + 13*scale, hip_y],
        [hip_x - 11*scale, hip_y],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [torso_pts], (*HOODIE_BASE.astype(int).tolist(), 255), lineType=cv2.LINE_AA)

    bp_pts = np.array([
        [shoulder_x + 8*scale, shoulder_y + 4*scale],
        [shoulder_x + 25*scale, shoulder_y + 11*scale],
        [shoulder_x + 23*scale, shoulder_y + 44*scale],
        [hip_x + 10*scale, hip_y - 2*scale],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [bp_pts], (*BACKPACK_COL.astype(int).tolist(), 255), lineType=cv2.LINE_AA)

    # 4. 머리
    neck_x = shoulder_x - 3*scale
    neck_y = shoulder_y - 8*scale
    cv2.circle(canvas, (int(neck_x), int(neck_y)), int(6.5 * scale), (*SKIN_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)
    head_x = neck_x - 5*scale
    head_y = neck_y - 17*scale
    cv2.ellipse(canvas, (int(head_x), int(head_y)), (int(11*scale), int(15*scale)), -6, 0, 360, (*SKIN_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(head_x + 1*scale), int(head_y - 4*scale)), (int(11.5*scale), int(12.5*scale)), -8, 150, 360, (*HAIR_COL.astype(int).tolist(), 255), -1, cv2.LINE_AA)

    # 5. 가까운 쪽 다리
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

    # 6. 가까운 쪽 팔
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

def blend_student_onto_frame(frame, ground_x, ground_y, phase, scale=1.0):
    sprite, feet_cx, feet_gy = render_realistic_student(phase, scale=scale)
    sw, sh = sprite.shape[1], sprite.shape[0]

    # 바닥 그림자
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
# 2. 모든 착석 학생 다중 레이어 자연 모션 합성 (Organic Kinematic Deformation)
# ==============================================================================
def apply_all_students_natural_motion(base_img, t):
    """
    모든 학생들이 가만히 있지 않고 살아 숨쉬며 작업하는 자연스러운 미세 움직임 합성.
    각 학생마다 독립적인 호흡 주기, 시선 이동, 손 조작(마우스, 타이핑, 필기)을 적용하여
    자연스러운 실사 움직임을 연출합니다.
    """
    frame = base_img.copy()
    h, w = frame.shape[:2]

    # 학생별 세부 모션 설정 (상대 좌표 rel_cx, rel_cy, 영향반경 rx, ry, dx, dy)
    student_motion_defs = [
        # [앞줄 학생 #1: 회색 반팔, 안경] 마우스 조작, 화면 주시 고개 숙임 및 호흡
        {
            "id": "Student #1",
            "bbox": (462, 435, 530, 520),
            "motions": [
                # 머리: 화면과 키보드 교차 주시 (부드러운 피치 & 요)
                (0.50, 0.25, 0.35, 0.28, 2.2 * math.sin(t * 1.8), 1.8 * math.cos(t * 1.4)),
                # 상체 호흡 및 어깨 미세 들썩임
                (0.50, 0.65, 0.45, 0.35, 0.8 * math.sin(t * 0.9), 2.0 * math.sin(t * 1.7)),
                # 오른손 마우스 조작 모션 (마우스패드 위 작은 원호)
                (0.32, 0.85, 0.28, 0.22, 2.8 * math.sin(t * 3.5), 1.6 * math.cos(t * 4.2))
            ],
            "activity": "MOUSE ACTIVE",
            "conf": 0.96 + 0.02 * math.sin(t * 2.1)
        },
        # [앞줄 학생 #2: 네이비 후드티] 양손 빠른 키보드 타이핑, 고개 좌우 주시, 상체 리듬
        {
            "id": "Student #2",
            "bbox": (558, 452, 632, 555),
            "motions": [
                # 머리: 좌우 모니터 주시
                (0.50, 0.26, 0.38, 0.26, 2.5 * math.sin(t * 1.5 + 0.8), 2.0 * math.cos(t * 1.9 + 0.3)),
                # 상체 호흡 & 타이핑 반동
                (0.50, 0.65, 0.45, 0.35, 1.0 * math.cos(t * 1.1 + 0.5), 2.2 * math.sin(t * 1.8 + 0.4)),
                # 키보드 위 양손 타자 버스트 (타이핑 시 손가락 빠른 진동 + 간헐적 멈춤)
                (0.50, 0.88, 0.42, 0.18, 3.4 * math.sin(t * 8.5) * (0.5 + 0.5 * math.sin(t * 1.2)), 2.6 * math.cos(t * 7.8))
            ],
            "activity": "TYPING",
            "conf": 0.97 + 0.02 * math.cos(t * 1.8)
        },
        # [앞줄 학생 #3: 밝은 회색 후드티] 볼펜 필기 모션, 고개 숙여 시험지 응시 후 화면 확인
        {
            "id": "Student #3",
            "bbox": (674, 498, 750, 650),
            "motions": [
                # 머리: 시험지 필기 응시(아래) -> 모니터 확인(위)
                (0.50, 0.24, 0.36, 0.25, 2.8 * math.sin(t * 1.6 + 1.5), 2.4 * math.sin(t * 2.1 + 1.0)),
                # 상체 호흡
                (0.50, 0.62, 0.45, 0.35, 1.2 * math.sin(t * 1.0 + 1.2), 2.2 * math.sin(t * 1.6 + 1.2)),
                # 오른손 볼펜 필기 획 긋기 (Pen Writing Motion)
                (0.38, 0.85, 0.32, 0.20, 3.6 * math.sin(t * 6.2) * (0.6 + 0.4 * math.sin(t * 1.5)), 2.2 * math.cos(t * 5.4))
            ],
            "activity": "NOTE WRITING",
            "conf": 0.95 + 0.03 * math.sin(t * 1.4)
        },
        # [2열 좌측 학생] 모니터 주시 및 고개 끄덕임
        {
            "id": "Student #201",
            "bbox": (485, 362, 524, 435),
            "motions": [
                (0.50, 0.30, 0.35, 0.30, 1.8 * math.sin(t * 1.9 + 2.1), 1.5 * math.cos(t * 1.3 + 1.8)),
                (0.50, 0.70, 0.40, 0.30, 0.8 * math.sin(t * 1.1), 1.6 * math.sin(t * 1.9 + 0.5))
            ],
            "activity": "ATTENTIVE",
            "conf": 0.93 + 0.02 * math.sin(t * 2.5)
        },
        # [2열 중앙 학생 (안경)] 시선 이동 및 필기/작업
        {
            "id": "Student #202",
            "bbox": (550, 368, 590, 420),
            "motions": [
                (0.50, 0.30, 0.35, 0.30, 1.6 * math.cos(t * 1.4 + 0.4), 1.4 * math.sin(t * 1.7 + 2.0)),
                (0.50, 0.70, 0.40, 0.30, 0.7 * math.sin(t * 1.0 + 2.5), 1.5 * math.sin(t * 1.7 + 1.5))
            ],
            "activity": "ATTENTIVE",
            "conf": 0.94 + 0.02 * math.cos(t * 2.2)
        },
        # [2열 우측 남학생 (검은 후드)] 손 움직임, 키보드 조작
        {
            "id": "Student #203",
            "bbox": (618, 375, 664, 475),
            "motions": [
                (0.50, 0.28, 0.35, 0.28, 2.0 * math.sin(t * 1.7 + 0.9), 1.8 * math.cos(t * 1.5 + 2.2)),
                (0.50, 0.68, 0.42, 0.32, 0.9 * math.cos(t * 1.2 + 1.1), 1.8 * math.sin(t * 1.8 + 0.8)),
                (0.50, 0.88, 0.35, 0.20, 2.4 * math.sin(t * 7.2), 1.6 * math.cos(t * 6.5))
            ],
            "activity": "WORKING",
            "conf": 0.94 + 0.02 * math.sin(t * 1.9)
        },
        # [2열 여학생 (장발)] 고개 갸우뚱, 화면 관찰
        {
            "id": "Student #204",
            "bbox": (695, 402, 738, 485),
            "motions": [
                (0.50, 0.30, 0.35, 0.28, 1.8 * math.cos(t * 1.6 + 1.7), 1.6 * math.sin(t * 1.8 + 1.1)),
                (0.50, 0.70, 0.42, 0.32, 0.8 * math.sin(t * 1.0 + 0.7), 1.7 * math.sin(t * 1.7 + 2.3))
            ],
            "activity": "ATTENTIVE",
            "conf": 0.93 + 0.02 * math.cos(t * 1.6)
        },
        # [좌측 벽면 학생 1]
        {
            "id": "Student #W01",
            "bbox": (50, 412, 105, 545),
            "motions": [
                (0.50, 0.30, 0.35, 0.30, 1.8 * math.sin(t * 1.3 + 0.2), 1.5 * math.cos(t * 1.6 + 0.5)),
                (0.50, 0.70, 0.45, 0.35, 1.0 * math.cos(t * 1.0), 1.8 * math.sin(t * 1.8))
            ],
            "activity": "WORKING",
            "conf": 0.91 + 0.02 * math.sin(t * 1.5)
        },
        # [좌측 벽면 학생 2]
        {
            "id": "Student #W02",
            "bbox": (110, 392, 160, 515),
            "motions": [
                (0.50, 0.30, 0.35, 0.30, 1.7 * math.cos(t * 1.5 + 1.2), 1.4 * math.sin(t * 1.4 + 0.8)),
                (0.50, 0.70, 0.45, 0.35, 0.9 * math.sin(t * 1.1 + 1.4), 1.7 * math.sin(t * 1.9 + 1.0))
            ],
            "activity": "WORKING",
            "conf": 0.92 + 0.02 * math.cos(t * 1.7)
        }
    ]

    # 각 학생 영역 패치별 가우시안 변형 적용 및 동적 바운딩 박스 계산
    dynamic_students = []
    pad = 18

    for s_info in student_motion_defs:
        x1, y1, x2, y2 = s_info["bbox"]
        px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
        px2, py2 = min(w, x2 + pad), min(h, y2 + pad)

        sub = base_img[py1:py2, px1:px2]
        sh, sw = sub.shape[:2]

        grid_y, grid_x = np.mgrid[0:sh, 0:sw].astype(np.float32)
        disp_x = np.zeros((sh, sw), dtype=np.float32)
        disp_y = np.zeros((sh, sw), dtype=np.float32)

        tot_dx, tot_dy = 0.0, 0.0
        for (rel_cx, rel_cy, rx, ry, dx, dy) in s_info["motions"]:
            mcx = rel_cx * sw
            mcy = rel_cy * sh
            mrx = rx * sw
            mry = ry * sh
            dist_sq = ((grid_x - mcx) / max(1.0, mrx)) ** 2 + ((grid_y - mcy) / max(1.0, mry)) ** 2
            w_gauss = np.exp(-dist_sq * 1.8)
            w_gauss[dist_sq > 3.0] = 0
            disp_x += w_gauss * dx
            disp_y += w_gauss * dy
            tot_dx += dx * 0.35
            tot_dy += dy * 0.35

        map_x = np.clip(grid_x + disp_x, 0, sw - 1)
        map_y = np.clip(grid_y + disp_y, 0, sh - 1)
        warped = cv2.remap(sub, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame[py1:py2, px1:px2] = warped

        # 움직임에 정확히 밀착 추적하는 다이내믹 바운딩 박스
        dyn_x1 = int(np.clip(x1 + tot_dx, 0, w - 1))
        dyn_y1 = int(np.clip(y1 + tot_dy, 0, h - 1))
        dyn_x2 = int(np.clip(x2 + tot_dx, 0, w - 1))
        dyn_y2 = int(np.clip(y2 + tot_dy, 0, h - 1))

        dynamic_students.append({
            "id": s_info["id"],
            "bbox": (dyn_x1, dyn_y1, dyn_x2, dyn_y2),
            "activity": s_info["activity"],
            "conf": s_info["conf"]
        })

    # 학생 #3의 볼펜 실시간 필기 변위 계산 (손 움직임과 100% 동기화)
    pen_dx = 3.6 * math.sin(t * 6.2) * (0.6 + 0.4 * math.sin(t * 1.5))
    pen_dy = 2.2 * math.cos(t * 5.4)

    return frame, dynamic_students, (pen_dx, pen_dy)


# ==============================================================================
# 3. 전체 파이프라인 생성 및 비디오 인코딩
# ==============================================================================
def generate_natural_multi_student_surveillance():
    print("[CCTV AI Surveillance] Generating multi-student natural motion surveillance video...")
    base_img = cv2.imread(str(IMG_SRC))
    if base_img is None:
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {IMG_SRC}")

    h, w, _ = base_img.shape
    fps = 20
    total_seconds = 12
    total_frames = fps * total_seconds

    # 상단 HUD 포함 전체 프레임 크기 (w, h + 80)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_multi = cv2.VideoWriter(str(VIDEO_OUT_MULTI), fourcc, fps, (w, h + 80))
    out_entry = cv2.VideoWriter(str(VIDEO_OUT_ENTRY), fourcc, fps, (w, h + 80))

    # 주요 감시 구역 (ROI)
    DOOR_ROI = (1030, 270, 1150, 620)
    path_points = [
        (1030, 520), # 출입문 문턱
        (1005, 555), # 복도 첫 걸음
        (960, 605),  # 중앙 복도
        (910, 660),  # 앞줄 통로
        (880, 625),  # 책상 방향 전환
        (870, 560)   # 책상 도착 및 착석
    ]

    # 정적 공용 기자재 (교탁 및 책상 장비)
    public_assets = [
        ("Public Monitor", (268, 565, 345, 725), (0, 255, 0)),
        ("Public PC", (218, 600, 272, 720), (0, 255, 0)),
        ("Public Keyboard", (215, 720, 300, 808), (0, 255, 0)),
        ("Public Mouse [SAFE]", (278, 796, 305, 828), (0, 0, 255)),
        ("Public Cable", (338, 706, 398, 786), (0, 255, 255)),
        ("Mug/Cup (Item)", (205, 688, 230, 728), (0, 100, 255)),
        # 학생 모니터들
        ("Monitor #101", (408, 472, 466, 565), (255, 215, 0)),
        ("Monitor #102", (504, 505, 578, 626), (255, 215, 0)),
        ("Monitor #103", (628, 555, 720, 690), (255, 215, 0)),
        ("Monitor #201", (515, 395, 555, 460), (255, 215, 0)),
        ("Monitor #202", (570, 408, 620, 475), (255, 215, 0)),
        ("Monitor #203", (650, 425, 710, 500), (255, 215, 0)),
        ("Monitor #204", (740, 448, 810, 530), (255, 215, 0)),
        ("Monitor #W01", (104, 392, 142, 442), (255, 215, 0))
    ]

    keyframes = {}
    motion_history = []

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # 1. 모든 학생들의 유기적 생체/작업 모션 적용된 베이스 프레임 렌더링
        frame, dynamic_students, (pen_dx, pen_dy) = apply_all_students_natural_motion(base_img, t)

        # 2. 공용 기자재 정밀 바운딩 박스 렌더링
        for label, (ax1, ay1, ax2, ay2), col in public_assets:
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), col, 1)
            cv2.putText(frame, label, (ax1 - 10, ay1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # 3. 학생 #3 책상 위 개인 소지품 (시험지, 실시간 필기 볼펜, 바닥 백팩)
        # 시험지
        cv2.rectangle(frame, (710, 648), (760, 692), (0, 100, 255), 2)
        cv2.putText(frame, "Exam Paper", (700, 642), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 100, 255), 1)

        # 볼펜 (손 움직임에 따라 실시간 정밀 연동!)
        dyn_pen_x1 = int(722 + pen_dx)
        dyn_pen_y1 = int(662 + pen_dy)
        dyn_pen_x2 = int(745 + pen_dx)
        dyn_pen_y2 = int(678 + pen_dy)
        cv2.rectangle(frame, (dyn_pen_x1, dyn_pen_y1), (dyn_pen_x2, dyn_pen_y2), (0, 0, 255), 2)
        cv2.putText(frame, "Pen [WRITING]", (dyn_pen_x1 - 25, dyn_pen_y2 + 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)

        # 학생 #3 바닥 백팩
        cv2.rectangle(frame, (712, 735), (775, 875), (0, 165, 255), 2)
        cv2.putText(frame, "Backpack [Personal]", (680, 730), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)

        # 4. 모든 착석 학생 다이내믹 AI 바운딩 박스 및 실시간 상태 태그
        for s in dynamic_students:
            bx1, by1, bx2, by2 = s["bbox"]
            label_text = f"{s['id']} [{s['activity']}] ({s['conf']:.2f})"

            # 앞줄 학생은 더 굵고 선명하게
            is_front = s["id"] in ["Student #1", "Student #2", "Student #3"]
            box_col = (255, 140, 0) if is_front else (220, 160, 50)
            thickness = 2 if is_front else 1

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_col, thickness)

            # 태그 배지
            cv2.putText(frame, label_text, (bx1 - 15, by1 - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41 if is_front else 0.35, box_col, 1)

        # ------------------------------------------------------------------
        # 단계별 시나리오 (출입문 진입 학생 파이프라인)
        # ------------------------------------------------------------------
        # [단계 1] (0.0s ~ 3.0s): 실습실 전원 정상 수업/작업 중 & 출입문 상시 감시
        if t < 3.0:
            phase_name = "PHASE 1: MULTI-STUDENT ACTIVE WORK & DOORWAY ARMED"
            sub_text = "All Students Naturally Active (Typing/Writing/Mouse) | Doorway ROI: CLEAR | Zero Inactive Figures"
            hud_color = (0, 255, 128)

            cv2.rectangle(frame, (DOOR_ROI[0], DOOR_ROI[1]), (DOOR_ROI[2], DOOR_ROI[3]), (0, 255, 0), 2)
            cv2.putText(frame, "ROI 2: ENTRY DOOR [CLEAR]", (DOOR_ROI[0] - 80, DOOR_ROI[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            if frame_idx == 15:
                keyframes["1_multi_student_active"] = frame.copy()

        # [단계 2] (3.0s ~ 5.5s): 실제 학생이 문을 열고 자연스럽게 첫 걸음을 딛는 순간
        elif 3.0 <= t < 5.5:
            phase_name = "PHASE 2: NATURAL WALKING ENTRY DETECTED"
            sub_text = ">> EVENT: Student Entering Doorway | Natural Gait Motion | Track ID #101 (Conf: 0.95) <<"
            hud_color = (0, 165, 255)

            cv2.rectangle(frame, (DOOR_ROI[0], DOOR_ROI[1]), (DOOR_ROI[2], DOOR_ROI[3]), (0, 0, 255), 2)
            cv2.putText(frame, "ROI 2: DOOR ENTRY EVENT TRIGGERED!", (DOOR_ROI[0] - 110, DOOR_ROI[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

            entry_progress = (t - 3.0) / 2.5
            p_gx = int(1030 * (1 - entry_progress) + 1005 * entry_progress)
            p_gy = int(520 * (1 - entry_progress) + 555 * entry_progress)

            walk_phase = (t - 3.0) * 6.2
            current_scale = 0.82 + 0.05 * entry_progress

            bbox = blend_student_onto_frame(frame, p_gx, p_gy, phase=walk_phase, scale=current_scale)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 215, 255), 2)
            cv2.putText(frame, "Person #101 [ENTRY] (0.95)", (bbox[0] - 25, bbox[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 215, 255), 2)

            if frame_idx == int(4.2 * fps):
                keyframes["2_person_entry"] = frame.copy()

        # [단계 3] (5.5s ~ 8.5s): 복도를 따라 다리를 교차하며 걸어오는 전신 추적
        elif 5.5 <= t < 8.5:
            phase_name = "PHASE 3: AISLE TRAJECTORY & ARTICULATED GAIT TRACKING"
            sub_text = "Tracking ID #101: Walking Down Corridor -> Workstation | All Surrounding Students Working"
            hud_color = (255, 200, 0)

            progress = (t - 5.5) / 3.0
            idx = min(int(progress * (len(path_points) - 1)), len(path_points) - 2)
            local_alpha = (progress * (len(path_points) - 1)) - idx
            cx = int(path_points[idx][0] * (1 - local_alpha) + path_points[idx+1][0] * local_alpha)
            cy = int(path_points[idx][1] * (1 - local_alpha) + path_points[idx+1][1] * local_alpha)

            motion_history.append((cx, cy))

            # 바닥 이동 궤적 선
            for i in range(len(motion_history) - 1):
                cv2.line(frame, motion_history[i], motion_history[i+1], (0, 240, 255), 2, cv2.LINE_AA)
            for pt in motion_history[::4]:
                cv2.circle(frame, pt, 3, (0, 180, 255), -1)

            walk_phase = (t - 5.5) * 6.5
            current_scale = 0.87 + 0.18 * progress

            bbox = blend_student_onto_frame(frame, cx, cy, phase=walk_phase, scale=current_scale)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 215, 255), 2)
            cv2.putText(frame, "Person #101 [WALKING GAIT] (0.96)", (bbox[0] - 25, bbox[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 215, 255), 2)

            if frame_idx == int(7.0 * fps):
                keyframes["3_moving_to_desk"] = frame.copy()

        # [단계 4] (8.5s ~ 12.0s): 책상 착석, 가방에서 마우스 꺼냄 & 실시간 AI 인식 연동
        else:
            phase_name = "PHASE 4: WORKSTATION SEATING & PERSONAL MOUSE ACQUIRED"
            sub_text = "Action: Seated -> Personal Mouse Unpacked | Confirmed Ownership Registration (Zero Blindspot)"
            hud_color = (0, 255, 200)

            # 전체 보행 궤적 유지
            for i in range(len(path_points) - 1):
                cv2.line(frame, path_points[i], path_points[i+1], (0, 240, 255), 2, cv2.LINE_AA)

            # 착석한 학생 #101 상체 바운딩 박스
            s101_breath = 2.0 * math.sin(t * 1.8)
            cv2.rectangle(frame, (810, int(480 + s101_breath)), (925, int(650 + s101_breath)), (255, 140, 0), 2)
            cv2.putText(frame, "Student #101 [SEATED & ACTIVE] (0.96)", (760, int(472 + s101_breath)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 140, 0), 2)

            # 바닥에 내려놓은 백팩
            cv2.rectangle(frame, (855, 715), (945, 865), (0, 165, 255), 2)
            cv2.putText(frame, "Backpack #101 [Floor]", (825, 708), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)

            # 책상 위 꺼내진 마우스 실시간 정밀 인식 (책상 상판 위)
            desk_m = (835, 560, 875, 605)
            cv2.rectangle(frame, (desk_m[0], desk_m[1]), (desk_m[2], desk_m[3]), (0, 255, 128), 2)
            cv2.putText(frame, "Personal Mouse [ACQUIRED] (0.97)", (desk_m[0] - 80, desk_m[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 128), 2)

            # 소유권 연결 벡터선 (Student #101 <=> Personal Mouse)
            cv2.line(frame, (850, int(540 + s101_breath)), (desk_m[0] + 20, desk_m[1]), (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "Ownership Link", (790, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)

            if frame_idx == int(10.5 * fps):
                keyframes["4_item_acquired"] = frame.copy()

        # ------------------------------------------------------------------
        # 상단 실시간 모니터링 HUD 대시보드
        # ------------------------------------------------------------------
        hud = np.zeros((80, w, 3), dtype=np.uint8)
        hud[:] = (20, 20, 26)
        cv2.putText(hud, f"[{phase_name}]", (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.70, hud_color, 2)
        cv2.putText(hud, sub_text, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1)

        timer_str = f"TIME 00:{int(t):02d}s | CCTV CH 04 [20 FPS]"
        cv2.putText(hud, timer_str, (w - 280, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        combined = np.vstack([hud, frame])
        out_multi.write(combined)
        out_entry.write(combined)

    out_multi.release()
    out_entry.release()
    print(f"[Success] Multi-student natural motion surveillance video generated successfully!")
    print(f" - Output 1: {VIDEO_OUT_MULTI}")
    print(f" - Output 2: {VIDEO_OUT_ENTRY}")

    # 주요 4단계 스틸컷 이미지 저장 및 아티팩트 복사
    for k_name, k_img in keyframes.items():
        k_path = OUTPUT_DIR / f"flow_{k_name}.jpg"
        cv2.imwrite(str(k_path), k_img)
        print(f" - Keyframe saved: {k_path}")
        for art_d in [CURR_ART_DIR, PREV_ART_DIR]:
            shutil.copy(str(k_path), str(art_d / f"flow_{k_name}.jpg"))

    for art_d in [CURR_ART_DIR, PREV_ART_DIR]:
        shutil.copy(str(VIDEO_OUT_MULTI), str(art_d / "classroom_natural_multi_student_cctv.mp4"))
        shutil.copy(str(VIDEO_OUT_ENTRY), str(art_d / "classroom_person_entry_demo.mp4"))

    print(f"[Success] All videos and keyframes copied to artifacts directory!")

if __name__ == "__main__":
    generate_natural_multi_student_surveillance()
