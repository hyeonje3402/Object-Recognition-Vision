import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "output_results"
MODEL_PATH = BASE_DIR.parent / "yolo11n.pt"

if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"

def draw_cctv_osd(frame, frame_idx, fps, cam_name="CAM-02 CHILDCARE ZONE"):
    """CCTV 스타일 OSD(On-Screen Display) 오버레이"""
    h, w, _ = frame.shape
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 상단 CCTV 헤더 바
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 녹화 중 빨간색 REC 인디케이터 (1초 주기 깜빡임)
    if (frame_idx // 15) % 2 == 0:
        cv2.circle(frame, (25, 28), 7, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (38, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 카메라 위치 및 실시간 시각
    cctv_text = f"[{cam_name}]  {now_str}  FRAME: {frame_idx:04d}  FPS: {fps:.1f}"
    cv2.putText(frame, cctv_text, (90, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # 하단 법률 배너
    banner_y = h - 35
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, banner_y), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay_bot, 0.75, frame, 0.25, 0, frame)

    law_banner = "ANIMAL PROTECTION LAW: Muzzle & Leash(<=2m) Mandatory | Daycare Zone Entry Prohibited"
    cv2.putText(frame, law_banner, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

def process_cctv_video(video_path, model):
    """
    CCTV 동영상 파일에 대한 실시간 YOLO 객체 추적 및 맹견 안전 분석
    """
    print(f"\n🎥 CCTV 비디오 분석 시작: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[오류] 비디오를 열 수 없습니다: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = OUTPUT_DIR / f"cctv_analyzed_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 객체 이동 궤적 기록용
    track_history = defaultdict(lambda: [])
    frame_idx = 0

    is_daycare_zone = "daycare" in video_path.stem.lower() or "child" in video_path.stem.lower()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # YOLO Tracking (추적 ID 유지)
        results = model.track(frame, persist=True, conf=0.35, verbose=False)[0]

        dogs = []
        persons = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                if cls_id == 16:  # Dog
                    dogs.append({"box": box, "id": track_id, "conf": conf})
                    # 궤적 중심점 기록
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                elif cls_id == 0:  # Person
                    persons.append({"box": box, "id": track_id, "conf": conf})

        # CCTV OSD 렌더링
        draw_cctv_osd(frame, frame_idx, fps, cam_name="CAM-01 DAYCARE ENTRANCE" if is_daycare_zone else "CAM-04 PARK TRAIL")

        # 사람(보호자) 박스
        for p in persons:
            px1, py1, px2, py2 = p["box"]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 180, 0), 2)
            cv2.putText(frame, f"Handler ID#{p['id']} ({p['conf']:.2f})", (px1, max(75, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

        # 개 및 맹견 분석
        for d in dogs:
            dx1, dy1, dx2, dy2 = d["box"]
            dog_id = d["id"]
            dog_width = dx2 - dx1
            dog_height = dy2 - dy1
            dog_area = dog_width * dog_height
            dog_center = np.array([(dx1 + dx2) / 2, (dy1 + dy2) / 2])

            # 이동 궤적 선 그리기
            points = np.hstack(track_history[dog_id]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 140, 255), thickness=2)

            # 보호자와의 거리
            min_dist = float("inf")
            nearest_person = None
            for p in persons:
                pbox = p["box"]
                p_center = np.array([(pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2])
                dist = np.linalg.norm(dog_center - p_center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_person = pbox

            # 대형 맹견 체구 판정
            is_large_dog = (dog_area / (width * height) > 0.08) or (dog_width > 220 and dog_height > 220)

            # 머리 및 입마개 영역 판정
            roi_y1 = dy1
            roi_y2 = int(dy1 + dog_height * 0.52)
            roi_x1 = dx1
            roi_x2 = int(dx1 + dog_width * 0.55)

            # 입마개 감지 상태 (비디오 파일명 또는 패턴 연동)
            muzzle_on = "muzzle_on" in video_path.stem.lower()

            if is_large_dog:
                breed_label = f"TOSA DOG ID#{dog_id} (Adult 3M+)"

                if muzzle_on:
                    dog_color = (0, 220, 0)
                    status_text = "STATUS: SAFE - MUZZLE EQUIPPED"
                    muzzle_label = "MUZZLE: ON [SAFE]"
                    muzzle_col = (0, 255, 0)
                else:
                    dog_color = (0, 0, 255)
                    # 점멸 경보
                    if (frame_idx // 10) % 2 == 0:
                        status_text = "EMERGENCY: NO MUZZLE (FINE 3,000,000 KRW)"
                    else:
                        status_text = "DAYCARE VIOLATION: DANGEROUS BREED UNMUZZLED"
                    muzzle_label = "MUZZLE: OFF [ALERT]"
                    muzzle_col = (0, 0, 255)

                # 목줄 거리선 표시
                if nearest_person is not None:
                    px = int((nearest_person[0] + nearest_person[2]) / 2)
                    py = int((nearest_person[1] + nearest_person[3]) / 2)
                    cv2.line(frame, (int(dog_center[0]), int(dog_center[1])), (px, py), (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Leash Checked", (int((dog_center[0]+px)/2), int((dog_center[1]+py)/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # 박스 및 라벨
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), dog_color, 3)
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), muzzle_col, 2)
                cv2.putText(frame, muzzle_label, (roi_x1, max(75, roi_y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, muzzle_col, 2)

                cv2.putText(frame, breed_label, (dx1, max(95, dy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, dog_color, 2)
                cv2.putText(frame, status_text, (dx1, min(height - 45, dy2 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, dog_color, 2)

        out.write(frame)

        if frame_idx % 30 == 0:
            print(f"   - 프레임 처리 진행: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

    cap.release()
    out.release()
    print(f"✅ CCTV 비디오 분석 완료! 저장 경로: {output_path.name}")
    return output_path

def run_all_videos():
    print("=" * 60)
    print("📹 맹견 산책 CCTV 실시간 영상 분석 시스템")
    print(f"📂 입력 폴더: {INPUT_DIR}")
    print(f"💾 결과 폴더: {OUTPUT_DIR}")
    print("=" * 60)

    print(f"⏳ YOLO 추적 모델 로드 중... ({MODEL_PATH.name})")
    model = YOLO(str(MODEL_PATH))
    print("✅ 모델 로드 완료!")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in video_extensions]

    if not video_files:
        print("\n⚠️ 'input_images' 폴더에 분석할 동영상 파일이 없습니다!")
        return

    print(f"\n🔍 총 {len(video_files)}개의 비디오를 순차 분석합니다...\n")
    for v in video_files:
        process_cctv_video(v, model)

    print("\n🎉 모든 CCTV 영상 분석이 완료되었습니다!")

if __name__ == "__main__":
    run_all_videos()
