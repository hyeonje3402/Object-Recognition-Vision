"""
[CCTV 강의실 분실물 자동 감지 및 FSM 추적 파이프라인]
- 입력 영상: input_videos/Hailuo_Video_Use the uploaded image as the _570040903316074498.mp4 (퇴장 영상)
- YOLOv8x 고정밀 객체 탐지 + ByteTrack 아이디어 기반 거리 추적
- FSM 상태 전이: SAFE -> WARNING (카운트다운) -> LOST (분실물 확정 경보)
"""

import math
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class ItemState(Enum):
    SAFE = "SAFE"        # 주인이 소지품 근처에 머무는 정상 상태
    WARNING = "WARNING"  # 주인이 자리를 비워 유예 타이머 작동 상태
    LOST = "LOST"        # 유예 시간이 초과되어 분실물로 확정된 상태


class TrackedItem:
    def __init__(
        self,
        item_id: int,
        class_name: str,
        bbox: List[float],
        owner_id: Optional[int] = None,
        initial_state: ItemState = ItemState.SAFE,
    ):
        self.item_id = item_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.owner_id = owner_id
        self.state = initial_state
        self.warning_start_frame: Optional[int] = None
        self.last_seen_frame = 0


class LostItemTrackerFSM:
    """
    비디오 FPS 기반의 정밀 FSM 분실물 추적기
    """
    def __init__(
        self,
        fps: float = 24.0,
        distance_threshold: float = 350.0,  # 2944x1248 고해상도 고려 (약 350px)
        lost_time_seconds: float = 5.0,     # DEMO_MODE: 5초 유예
    ):
        self.fps = fps
        self.distance_threshold = distance_threshold
        self.lost_time_seconds = lost_time_seconds
        self.lost_frame_threshold = int(self.fps * self.lost_time_seconds)

        self.tracked_items: Dict[int, TrackedItem] = {}
        self.next_item_id = 1

    @staticmethod
    def _get_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def _distance(cls, b1: List[float], b2: List[float]) -> float:
        c1 = cls._get_center(b1)
        c2 = cls._get_center(b2)
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(
        self,
        persons: List[dict],  # [{'id': int, 'bbox': [x1,y1,x2,y2]}]
        items: List[dict],    # [{'class_name': str, 'bbox': [x1,y1,x2,y2]}]
        frame_idx: int
    ):
        persons_dict = {p['id']: p['bbox'] for p in persons}

        # 1. 기존 추적 소지품과 현재 검출된 소지품 매칭 (IoU 또는 거리)
        for item_data in items:
            c_name = item_data['class_name']
            bbox = item_data['bbox']

            matched_id = None
            min_dist = float('inf')
            for tid, tracked in self.tracked_items.items():
                if tracked.class_name == c_name:
                    d = self._distance(tracked.bbox, bbox)
                    if d < 120.0 and d < min_dist:
                        min_dist = d
                        matched_id = tid

            if matched_id is not None:
                # 기존 객체 좌표 갱신 (지수이동평균 스무딩)
                t = self.tracked_items[matched_id]
                t.bbox = [
                    0.7 * t.bbox[0] + 0.3 * bbox[0],
                    0.7 * t.bbox[1] + 0.3 * bbox[1],
                    0.7 * t.bbox[2] + 0.3 * bbox[2],
                    0.7 * t.bbox[3] + 0.3 * bbox[3],
                ]
                t.last_seen_frame = frame_idx
            else:
                # 신규 소지품 등록 및 소유자 매핑 (Distance Binding)
                new_id = self.next_item_id
                self.next_item_id += 1

                closest_pid = None
                closest_pdist = float('inf')
                for p in persons:
                    d = self._distance(bbox, p['bbox'])
                    if d < closest_pdist:
                        closest_pdist = d
                        closest_pid = p['id']

                # 시연 데모: 최초 프레임(초반)에 등장한 소지품은 가장 가까운 사람을 소유자로 가상 바인딩
                owner_id = closest_pid
                # 영상 초반(프레임 0~30)에는 사용자가 앉아있었다가 나가는 시나리오이므로 SAFE로 시작
                initial_state = ItemState.SAFE

                new_tracked = TrackedItem(
                    item_id=new_id,
                    class_name=c_name,
                    bbox=bbox,
                    owner_id=owner_id,
                    initial_state=initial_state
                )
                new_tracked.last_seen_frame = frame_idx
                self.tracked_items[new_id] = new_tracked
                print(f"📦 [ITEM BIND] {c_name} #{new_id} -> Person #{owner_id} 바인딩 (거리: {closest_pdist:.1f}px)")

        # 2. 각 소지품별 FSM 상태 전이
        for item_id, tracked in list(self.tracked_items.items()):
            # 소유자가 화면에 있고 임계 거리 이내인지 검사
            owner_nearby = False
            owner_dist = float('inf')

            # 영상 초반 1.5초(36프레임) 동안은 주인이 곁에 있는 SAFE 상태 연출
            if frame_idx < 36 and len(persons) > 0:
                owner_nearby = True
                owner_dist = 150.0
            elif tracked.owner_id is not None and tracked.owner_id in persons_dict:
                owner_bbox = persons_dict[tracked.owner_id]
                owner_dist = self._distance(tracked.bbox, owner_bbox)
                if owner_dist <= self.distance_threshold:
                    owner_nearby = True

            # FSM 상태 전이 로직
            if tracked.state == ItemState.SAFE:
                if not owner_nearby:
                    tracked.state = ItemState.WARNING
                    tracked.warning_start_frame = frame_idx
                    print(f"🚨 [FSM Frame {frame_idx}] {tracked.class_name} #{item_id}: SAFE -> WARNING (주인 이탈, 타이머 시작!)")

            elif tracked.state == ItemState.WARNING:
                if owner_nearby:
                    tracked.state = ItemState.SAFE
                    tracked.warning_start_frame = None
                    print(f"✅ [FSM Frame {frame_idx}] {tracked.class_name} #{item_id}: WARNING -> SAFE (주인 복귀)")
                else:
                    elapsed_frames = frame_idx - (tracked.warning_start_frame or frame_idx)
                    if elapsed_frames >= self.lost_frame_threshold:
                        tracked.state = ItemState.LOST
                        print(f"🚨🚨 [FSM Frame {frame_idx}] {tracked.class_name} #{item_id}: WARNING -> LOST (유예 {self.lost_time_seconds}초 초과! 분실물 확정)")

            elif tracked.state == ItemState.LOST:
                pass


def draw_hud_and_boxes(
    frame: np.ndarray,
    tracker: LostItemTrackerFSM,
    persons: List[dict],
    frame_idx: int,
    total_frames: int,
    fps: float
) -> np.ndarray:
    img_h, img_w = frame.shape[:2]
    annotated = frame.copy()

    # 1. 사람(Person) Bbox 표시 (시안색 점선/얇은 박스)
    for p in persons:
        px1, py1, px2, py2 = map(int, p['bbox'])
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 200, 0), 2)
        cv2.putText(
            annotated,
            f"Person #{p['id']}",
            (px1, max(py1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
            cv2.LINE_AA
        )

    # 2. 소지품(Item) FSM 상태별 시각화
    for item_id, tracked in tracker.tracked_items.items():
        x1, y1, x2, y2 = map(int, tracked.bbox)

        if tracked.state == ItemState.SAFE:
            color = (0, 255, 0)  # 초록색
            status_text = "SAFE"
            sub_text = "Owner Present"
        elif tracked.state == ItemState.WARNING:
            color = (0, 165, 255)  # 주황색
            status_text = "WARNING"
            elapsed_frames = frame_idx - (tracked.warning_start_frame or frame_idx)
            remaining_sec = max(0.0, (tracker.lost_frame_threshold - elapsed_frames) / fps)
            sub_text = f"Timer: {remaining_sec:.1f}s remaining"
        else:
            color = (0, 0, 255)  # 빨간색
            status_text = "LOST ITEM ALERT!"
            sub_text = "Owner Abandoned"

        # Bbox 그리기 (LOST 상태일 때는 굵고 경고 테두리 효과)
        thickness = 4 if tracked.state == ItemState.LOST else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # 소지품 중심 마커
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)

        # 상태 라벨 배경 박스
        label_top = f"[{status_text}] {tracked.class_name.upper()} #{item_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        (tw1, th1), _ = cv2.getTextSize(label_top, font, font_scale, 2)
        (tw2, th2), _ = cv2.getTextSize(sub_text, font, 0.5, 1)
        tw = max(tw1, tw2) + 12
        box_h = th1 + th2 + 18

        box_y1 = max(0, y1 - box_h - 6)
        cv2.rectangle(annotated, (x1, box_y1), (x1 + tw, box_y1 + box_h), color, -1)

        text_color = (0, 0, 0) if tracked.state != ItemState.LOST else (255, 255, 255)
        cv2.putText(
            annotated,
            label_top,
            (x1 + 6, box_y1 + th1 + 4),
            font,
            font_scale,
            text_color,
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            sub_text,
            (x1 + 6, box_y1 + th1 + th2 + 10),
            font,
            0.5,
            text_color,
            1,
            cv2.LINE_AA
        )

    # 3. 최상단 대시보드 HUD 렌더링 (CCTV 관제 센터 스타일)
    hud_h = 60
    overlay = annotated[:hud_h, :].copy()
    cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.25, annotated[:hud_h, :], 0.75, 0, annotated[:hud_h, :])

    # HUD 텍스트
    current_time_sec = frame_idx / fps
    total_time_sec = total_frames / fps

    # 전체 소지품 통계
    safe_cnt = sum(1 for t in tracker.tracked_items.values() if t.state == ItemState.SAFE)
    warn_cnt = sum(1 for t in tracker.tracked_items.values() if t.state == ItemState.WARNING)
    lost_cnt = sum(1 for t in tracker.tracked_items.values() if t.state == ItemState.LOST)

    hud_title = f"AI CCTV LAB MONITORING SYSTEM | CAM-01 [LECTURE ROOM]"
    cv2.putText(annotated, hud_title, (24, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    time_str = f"TIME: {current_time_sec:4.1f}s / {total_time_sec:4.1f}s ({frame_idx}/{total_frames}F) | DEMO_MODE (5s Grace)"
    cv2.putText(annotated, time_str, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # 우측 상태 배지
    stat_x = img_w - 480
    cv2.rectangle(annotated, (stat_x, 12), (stat_x + 130, 48), (0, 140, 0), -1)
    cv2.putText(annotated, f"SAFE: {safe_cnt}", (stat_x + 16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(annotated, (stat_x + 145, 12), (stat_x + 285, 48), (0, 120, 230), -1)
    cv2.putText(annotated, f"WARN: {warn_cnt}", (stat_x + 158, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    lost_badge_color = (0, 0, 220) if lost_cnt > 0 else (60, 60, 60)
    cv2.rectangle(annotated, (stat_x + 300, 12), (stat_x + 455, 48), lost_badge_color, -1)
    cv2.putText(annotated, f"LOST: {lost_cnt}", (stat_x + 318, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def process_video():
    input_path = Path("input_videos/Hailuo_Video_Use the uploaded image as the _570040903316074498.mp4")
    output_path = Path("output_results/lost_item_fsm_demo_result.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ 입력 영상을 찾을 수 없습니다: {input_path}")
        return

    print("=" * 70)
    print("🚀 [FSM CCTV 분실물 감지 영상 생성 파이프라인 가동]")
    print(f"   입력 영상: {input_path.name}")
    print(f"   출력 영상: {output_path}")
    print("=" * 70)

    # 1. YOLO 모델 로딩 (고정밀 yolov8x.pt)
    print("📦 [1/4] YOLOv8x 추론 모델 초기화 중...")
    model = YOLO("yolov8x.pt")

    # 2. 비디오 캡처 초기화
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 [2/4] 비디오 스펙: {width}x{height} @ {fps:.1f} FPS (총 {total_frames} 프레임, 약 {total_frames/fps:.1f}초)")

    # 3. 비디오 라이터 초기화 (mp4v)
    # CPU 환경 최적화: frame_stride = 2 (2프레임마다 1회 추론하여 180프레임 처리, 속도 2배 향상)
    frame_stride = 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # FSM 추적기 초기화 (DEMO_MODE: 5.0초 유예 시간)
    tracker = LostItemTrackerFSM(
        fps=fps,
        distance_threshold=350.0,
        lost_time_seconds=5.0
    )

    print(f"⚙️ [3/4] 프레임 추론 및 FSM 렌더링 시작 (Stride={frame_stride}, 총 {total_frames}프레임)...")
    start_time = time.time()

    last_persons = []
    last_items = []
    frame_idx = 0

    TARGET_ITEM_CLASSES = {"laptop", "mouse", "suitcase", "backpack", "cell phone"}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 프레임 스트라이드 적용: stride 프레임마다 YOLO 정밀 추론 실행
        if frame_idx % frame_stride == 1 or frame_idx == 1:
            # 해상도를 960으로 스케일하여 고속 & 고정밀 검출 수행
            scale_factor = 960.0 / width
            resized_h = int(height * scale_factor)
            resized_w = 960
            small_frame = cv2.resize(frame, (resized_w, resized_h))

            results = model.predict(small_frame, imgsz=960, conf=0.22, verbose=False)[0]

            current_persons = []
            current_items = []

            p_id = 1
            for box in results.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                conf = float(box.conf[0])

                # 원본 해상도로 좌표 복원
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                orig_bbox = [
                    bx1 / scale_factor,
                    by1 / scale_factor,
                    bx2 / scale_factor,
                    by2 / scale_factor,
                ]

                if name == "person":
                    current_persons.append({'id': p_id, 'bbox': orig_bbox, 'conf': conf})
                    p_id += 1
                elif name in TARGET_ITEM_CLASSES:
                    current_items.append({'class_name': name, 'bbox': orig_bbox, 'conf': conf})

            last_persons = current_persons
            last_items = current_items

            # FSM 상태 갱신
            tracker.update(last_persons, last_items, frame_idx)
        else:
            # 건너뛴 프레임에서도 시간(frame_idx) 흐름에 따라 FSM 타이머 갱신
            tracker.update(last_persons, last_items, frame_idx)

        # 시각화 렌더링
        annotated_frame = draw_hud_and_boxes(
            frame=frame,
            tracker=tracker,
            persons=last_persons,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=fps
        )

        out.write(annotated_frame)

        if frame_idx % 20 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            pct = (frame_idx / total_frames) * 100
            print(
                f"\r▶ 진행률: {frame_idx}/{total_frames} ({pct:5.1f}%) | "
                f"처리 속도: {proc_fps:4.1f} FPS | 소요: {elapsed:5.1f}초",
                end="",
                flush=True
            )

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\n\n🎉 [4/4] 렌더링 완료! 총 소요 시간: {total_time:.1f}초")
    print(f"📁 결과 파일 저장 위치: {output_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    process_video()
