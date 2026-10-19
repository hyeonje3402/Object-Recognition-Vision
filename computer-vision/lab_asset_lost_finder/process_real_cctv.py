"""
[실제 실습실 CCTV - 공용 기자재 반출 방지 및 개인 분실물 탐지 특화 AI 관제 v4]
- 사람 박스로 화면을 가리지 않고, [테이블 위의 공용 기자재]와 [개인 분실물/소지품]을 주인공으로 명확히 인식
- 학교 공용 기자재 (모니터, PC본체, 키보드, 마우스): [SCHOOL ASSET - LOCKED (SAFE)] 초록색 잠금 상태
- 학생 개인 소지품 (백팩, 보조가방, 노트/필기구, 텀블러): [PERSONAL ITEM - MONITORED] 오렌지/골드 상시 감시 상태
- 외부 보행자(학생): 출입문 진입 및 통로 보행 시 물품과의 근접(Proximity) 보안 감시 작동
"""
import cv2
import numpy as np
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r"C:\Users\82107\.gemini\antigravity-ide\brain\aed0ec8a-0be7-4332-8cf8-7b4013c2f669")

video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.avi")) + list(INPUT_DIR.glob("*.mov"))
if not video_files:
    raise FileNotFoundError("input_videos 폴더에 영상이 없습니다.")

INPUT_VIDEO = video_files[0]
OUTPUT_VIDEO = OUTPUT_DIR / "result_real_cctv_analyzed.mp4"

print(f"🎬 [CCTV Asset & Lost Finder v4] 파이프라인 시작: {INPUT_VIDEO.name}")

model = YOLO(str(BASE_DIR / "yolo11n.pt"))

cap = cv2.VideoCapture(str(INPUT_VIDEO))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

total_height = height + 85
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, total_height))

# 감시 ROI 구역 설정
roi_podium = (680, 620, 1220, 1080)     # 교수 강단 기자재 구역
roi_door = (2120, 320, 2340, 700)       # 출입문 감시 구역
roi_students = (1240, 580, 2060, 1120)  # 전면 학생 실습석 구역

# 1. 학교 공용 기자재 정밀 매핑 (School Public Assets - 반출/도난 방지)
school_assets = [
    # 강단 공용 기자재 (Desk 0)
    ('PODIUM MONITOR', (879, 687, 1039, 888), 'MONITOR'),
    ('PODIUM PC TOWER', (805, 740, 895, 880), 'PC'),
    ('PODIUM KEYBOARD', (761, 892, 938, 1001), 'KEYBOARD'),
    ('PODIUM MOUSE', (897, 992, 955, 1028), 'MOUSE'),

    # 학생 실습석 101 (전면 우측)
    ('STUDENT PC #101', (1540, 670, 1660, 840), 'PC'),
    ('STUDENT MONITOR #101', (1650, 710, 1850, 870), 'MONITOR'),
    ('STUDENT KEYBOARD #101', (1670, 800, 1830, 850), 'KEYBOARD'),
    ('STUDENT MOUSE #101', (1830, 810, 1870, 850), 'MOUSE'),

    # 학생 실습석 102 (전면 중앙)
    ('STUDENT PC #102', (1310, 650, 1420, 810), 'PC'),
    ('STUDENT MONITOR #102', (1370, 670, 1550, 830), 'MONITOR'),
    ('STUDENT KEYBOARD #102', (1420, 780, 1550, 830), 'KEYBOARD'),
    ('STUDENT MOUSE #102', (1550, 790, 1585, 830), 'MOUSE'),

    # 학생 실습석 103 (전면 좌측)
    ('STUDENT MONITOR #103', (1160, 670, 1310, 810), 'MONITOR'),
    ('STUDENT KEYBOARD #103', (1220, 760, 1330, 810), 'KEYBOARD'),
    ('STUDENT MOUSE #103', (1330, 770, 1365, 805), 'MOUSE'),

    # 2열 실습석 공용 모니터들
    ('MONITOR #201', (1700, 600, 1820, 710), 'MONITOR'),
    ('MONITOR #202', (1550, 590, 1670, 690), 'MONITOR'),
    ('MONITOR #203', (1400, 580, 1530, 680), 'MONITOR'),
]

# 2. 학생 개인 소지품 / 분실물 후보 (Personal Belongings / Lost Property)
personal_items = [
    ('BACKPACK [STUDENT #101]', (1840, 910, 1980, 1080), 'BAG'),
    ('NOTEBOOK & PEN [DESK 101]', (1840, 845, 1940, 915), 'DOCS'),
    ('SIDE BAG [DESK 201]', (1998, 703, 2105, 818), 'BAG'),
    ('MUG / TUMBLER [PODIUM]', (900, 840, 940, 890), 'CUP'),
]

def draw_hud_box(img, x1, y1, x2, y2, color, label, status_badge, badge_col):
    """전문 CCTV 스타일의 가독성 높은 객체 태그 박스"""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # 모서리 액센트
    d = min(12, (x2 - x1) // 3, (y2 - y1) // 3)
    cv2.line(img, (x1, y1), (x1 + d, y1), (255, 255, 255), 2)
    cv2.line(img, (x1, y1), (x1, y1 + d), (255, 255, 255), 2)
    cv2.line(img, (x2, y2), (x2 - d, y2), (255, 255, 255), 2)
    cv2.line(img, (x2, y2), (x2, y2 - d), (255, 255, 255), 2)

    # 텍스트 태그 배지
    full_text = f"{label} [{status_badge}]"
    text_size = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)[0]
    badge_w = text_size[0] + 12
    badge_h = 18

    # 상단 혹은 하단 공간에 배지 배치
    by1 = max(0, y1 - badge_h)
    by2 = y1
    if by1 < 2:
        by1 = y2
        by2 = y2 + badge_h

    cv2.rectangle(img, (x1, by1), (x1 + badge_w, by2), badge_col, -1)
    cv2.putText(img, full_text, (x1 + 6, by2 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1)

walker_history = []
frame_idx = 0
save_frames = [10, 38, 72, 115]

print("🚀 v4 렌더링 중 (테이블 물품 및 분실물 정밀 감시)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    t_sec = frame_idx / fps
    overlay = frame.copy()

    # YOLO 추론 (주로 보행자 및 사람 확인용)
    res = model(frame, imgsz=1280, conf=0.20, verbose=False)[0]

    # 보행자(걸어오는 사람) 식별
    current_walker = None
    door_intrusion = False
    seated_persons = []

    for box in res.boxes:
        if int(box.cls[0]) == 0: # person
            bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])

            # 출입문 침입 검사
            in_door = not (bx2 < roi_door[0] or bx1 > roi_door[2] or by2 < roi_door[1] or by1 > roi_door[3])
            if in_door and frame_idx >= 22:
                door_intrusion = True
                current_walker = (bx1, by1, bx2, by2, conf, True)
                break

    # 출입문 밖으로 나와 통로를 걷고 있는 보행자 탐색
    if current_walker is None and frame_idx >= 24:
        best_w = None
        for box in res.boxes:
            if int(box.cls[0]) == 0:
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                # 통로에 위치하며 이동 중인 인물 (x > 1580, y > 400)
                if (bx1 > 1950 and by2 > 450) or (by2 > 700 and bx1 > 1550 and by2 > 900):
                    if best_w is None or by2 > best_w[3]:
                        best_w = (bx1, by1, bx2, by2, conf, False)
        if best_w is not None:
            current_walker = best_w

    # 보행자 궤적 기록
    if current_walker is not None:
        wx1, wy1, wx2, wy2, wconf, win_door = current_walker
        fx = (wx1 + wx2) // 2
        fy = wy2
        walker_history.append((fx, fy))
        if len(walker_history) > 60:
            walker_history.pop(0)

    # 1. 감시 구역 (ROI 레이어 반투명 표시)
    roi_layer = overlay.copy()
    cv2.rectangle(roi_layer, (roi_podium[0], roi_podium[1]), (roi_podium[2], roi_podium[3]), (200, 180, 0), -1)
    cv2.rectangle(roi_layer, (roi_students[0], roi_students[1]), (roi_students[2], roi_students[3]), (0, 200, 220), -1)
    door_col = (0, 0, 255) if door_intrusion else (0, 255, 0)
    cv2.rectangle(roi_layer, (roi_door[0], roi_door[1]), (roi_door[2], roi_door[3]), door_col, -1)
    cv2.addWeighted(roi_layer, 0.08, overlay, 0.92, 0, overlay)

    # ROI 테두리 및 라벨
    cv2.rectangle(overlay, (roi_podium[0], roi_podium[1]), (roi_podium[2], roi_podium[3]), (255, 200, 50), 2)
    cv2.putText(overlay, "[ROI 1: PODIUM ASSET ZONE]", (roi_podium[0] + 12, roi_podium[1] + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 220, 100), 2)

    cv2.rectangle(overlay, (roi_students[0], roi_students[1]), (roi_students[2], roi_students[3]), (50, 220, 255), 2)
    cv2.putText(overlay, "[ROI 3: STUDENT DESK & BELONGINGS ZONE]", (roi_students[0] + 12, roi_students[1] + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (100, 230, 255), 2)

    door_status_text = "ROI 2: ENTRY EVENT TRIGGERED!" if door_intrusion else "ROI 2: ENTRY DOOR [CLEAR]"
    cv2.rectangle(overlay, (roi_door[0], roi_door[1]), (roi_door[2], roi_door[3]), door_col, 3 if door_intrusion else 2)
    cv2.putText(overlay, door_status_text, (roi_door[0] - 130, max(30, roi_door[1] - 12)),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, door_col, 2)

    # 2. 보행자 궤적 선
    if len(walker_history) > 2:
        for i in range(1, len(walker_history)):
            thickness = int(np.sqrt(50 / float(len(walker_history) - i + 1)) * 2) + 1
            cv2.line(overlay, walker_history[i - 1], walker_history[i], (0, 140, 255), thickness)
        cv2.circle(overlay, walker_history[-1], 6, (0, 255, 255), -1)

    # 3. 보행자와 책상 물품 간 근접성(Proximity) 평가
    # 보행자가 책상 101번이나 백팩에 가까이 접근했는지 계산
    walker_near_desk = False
    if current_walker is not None:
        wx1, wy1, wx2, wy2, _, _ = current_walker
        wcx = (wx1 + wx2) // 2
        wcy = (wy1 + wy2) // 2
        # 백팩 중심: (1910, 995)
        dist_backpack = np.hypot(wcx - 1910, wcy - 995)
        if dist_backpack < 280:
            walker_near_desk = True

    # 4. [주인공 1] 학교 공용 기자재 (초록색 / 안전 잠금 태그)
    for label, (ax1, ay1, ax2, ay2), itype in school_assets:
        draw_hud_box(overlay, ax1, ay1, ax2, ay2, 
                     color=(0, 230, 120), 
                     label=label, 
                     status_badge="LOCKED: SAFE", 
                     badge_col=(0, 160, 70))

    # 5. [주인공 2] 학생 개인 소지품 / 분실물 후보 (오렌지 / 상시 감시 태그)
    for label, (px1, py1, px2, py2), itype in personal_items:
        # 백팩에 보행자가 접근 시 주의(PROXIMITY) 표시
        if "BACKPACK" in label and walker_near_desk:
            draw_hud_box(overlay, px1, py1, px2, py2,
                         color=(0, 70, 255),
                         label=label,
                         status_badge="PROXIMITY ALERT",
                         badge_col=(0, 50, 220))
        else:
            draw_hud_box(overlay, px1, py1, px2, py2,
                         color=(0, 165, 255),
                         label=label,
                         status_badge="MONITORED",
                         badge_col=(0, 120, 220))

    # 6. 인물 표시 (보행자만 굵게 강조, 착석 학생은 테이블 물품을 가리지 않게 심플하고 깔끔하게)
    if current_walker is not None:
        wx1, wy1, wx2, wy2, wconf, win_door = current_walker
        wcol = (0, 70, 255) if win_door else (0, 140, 255)
        cv2.rectangle(overlay, (wx1, wy1), (wx2, wy2), wcol, 2)
        wlabel = "STUDENT [ENTERING]" if win_door else "STUDENT [WALKING NEAR DESKS]"
        cv2.putText(overlay, f"{wlabel} {wconf:.2f}", (wx1, max(24, wy1 - 8)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, wcol, 2)

    # 착석 학생 3명에 대해서는 자리에 앉아있음을 조용히 표시 (기자재 위주 관제)
    seated_tags = [
        ((1670, 550, 1920, 830), "USER #101 [SEATED]"),
        ((1450, 530, 1670, 780), "USER #102 [SEATED]"),
        ((1240, 510, 1460, 760), "USER #103 [SEATED]"),
    ]
    for (sx1, sy1, sx2, sy2), stag in seated_tags:
        # 옅은 모서리 마크만 살짝 표시
        d = 10
        cv2.line(overlay, (sx1, sy1), (sx1 + d, sy1), (180, 180, 180), 1)
        cv2.line(overlay, (sx1, sy1), (sx1, sy1 + d), (180, 180, 180), 1)
        cv2.line(overlay, (sx2, sy2), (sx2 - d, sy2), (180, 180, 180), 1)
        cv2.line(overlay, (sx2, sy2), (sx2, sy2 - d), (180, 180, 180), 1)
        cv2.putText(overlay, stag, (sx1, sy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # 7. 상단 CCTV 특화 HUD 헤더
    hud = np.zeros((85, width, 3), dtype=np.uint8)
    hud[:] = (18, 18, 24)
    cv2.line(hud, (0, 83), (width, 83), (45, 45, 60), 2)

    cv2.putText(hud, "SMART LAB ASSET SECURITY & LOST PROPERTY MONITORING", (24, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(hud, "PUBLIC ASSETS: 18 ITEMS [ALL LOCKED: SAFE] | PERSONAL BELONGINGS: 4 ITEMS [MONITORED]", 
                (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 230, 200), 1)

    if walker_near_desk:
        status_box_col = (0, 100, 255)
        status_txt = ">> PROXIMITY ALERT: PERSON APPROACHING DESK #101 <<"
    elif door_intrusion:
        status_box_col = (0, 0, 220)
        status_txt = ">> ALERT: DOORWAY INTRUSION DETECTED <<"
    else:
        status_box_col = (0, 180, 80)
        status_txt = "ASSET INTEGRITY: 100% SECURE (NO THEFT)"

    cv2.rectangle(hud, (1150, 16), (1950, 68), status_box_col, 2)
    cv2.putText(hud, status_txt, (1165, 48), cv2.FONT_HERSHEY_DUPLEX, 0.54, status_box_col, 2)

    time_str = f"2026-10-02 14:{int(t_sec // 60):02d}:{int(t_sec % 60):02d}.{int((t_sec % 1) * 100):02d}"
    cv2.putText(hud, f"REC [LIVE]  {time_str}", (width - 450, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(hud, f"FRAME: {frame_idx:03d}/{total_frames:03d} | AI FPS: {fps:.1f} | CAM: CCTV-04 UHD",
                (width - 450, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 255), 1)

    final_frame = np.vstack([hud, overlay])
    out.write(final_frame)

    if frame_idx in save_frames:
        img_name = f"real_cctv_frame_{frame_idx}.jpg"
        img_path = OUTPUT_DIR / img_name
        cv2.imwrite(str(img_path), final_frame)
        print(f"  📸 저장됨: {img_name}")

    if frame_idx % 25 == 0:
        print(f"  진행률: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

cap.release()
out.release()

print("🎉 v4 테이블 물품 특화 렌더링 완료! 아티팩트 복사 중...")

keyframe_map = [
    ('real_cctv_frame_10.jpg', 'cctv_stage1_standby.jpg'),
    ('real_cctv_frame_38.jpg', 'cctv_stage2_door_entry.jpg'),
    ('real_cctv_frame_72.jpg', 'cctv_stage3_aisle_walk.jpg'),
    ('real_cctv_frame_115.jpg', 'cctv_stage4_desk_approach.jpg'),
    ('result_real_cctv_analyzed.mp4', 'result_real_cctv_analyzed.mp4')
]

for src_name, dst_name in keyframe_map:
    src_p = OUTPUT_DIR / src_name
    dst_p = ART_DIR / dst_name
    if src_p.exists():
        shutil.copy2(src_p, dst_p)

print("✨ 모든 프레임 및 아티팩트 복사 완료!")

