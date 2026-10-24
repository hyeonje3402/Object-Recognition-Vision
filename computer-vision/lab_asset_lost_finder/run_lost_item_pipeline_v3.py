"""
[CCTV 강의실 분실물 자동 감지 및 FSM 추적 파이프라인 V3 - FIXED]
- 착용 중인 가방(Worn Backpack) 오인식 제거
- 사람이 메고 퇴장한 객체가 화면에 유령 박스로 남는 현상 완벽 해결
- 실제로 책상 위에 계속 방치된 소지품(Laptop, Mouse, Tablet 등)만 깔끔하게 FSM 추적
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
        self.missed_frames = 0  # 연속 미검출 프레임 수


class LostItemTrackerFSMFixed:
    def __init__(
        self,
        fps: float = 24.0,
        distance_threshold: float = 400.0,  # 2944x1248 고해상도 기준
        lost_time_seconds: float = 5.0,     # DEMO_MODE: 5초 유예
        max_missed_frames: int = 10,        # 사람이 메고 나가서 10프레임 이상 안 보이면 추적 제거
    ):
        self.fps = fps
        self.distance_threshold = distance_threshold
        self.lost_time_seconds = lost_time_seconds
        self.lost_frame_threshold = int(self.fps * self.lost_time_seconds)
        self.max_missed_frames = max_missed_frames

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

    @staticmethod
    def _is_inside(inner_box: List[float], outer_box: List[float]) -> bool:
        """내부 박스의 중심점이 외부 박스(사람) 안에 있는지 검사"""
        cx = (inner_box[0] + inner_box[2]) / 2.0
        cy = (inner_box[1] + inner_box[3]) / 2.0
        ox1, oy1, ox2, oy2 = outer_box
        return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)

    def filter_worn_items(self, items: List[dict], persons: List[dict]) -> List[dict]:
        """
        사람이 등에 메거나 착용 중인 백팩/가방은 분실물 대상에서 제외
        """
        filtered = []
        for item in items:
            c_name = item['class_name']
            bbox = item['bbox']

            if c_name in {"backpack", "handbag", "suitcase"}:
                # 사람의 Bbox 안에 중심점이 있는 경우 -> 사람이 착용 중인 것으로 판정하고 무시
                is_worn = False
                for p in persons:
                    if self._is_inside(bbox, p['bbox']):
                        is_worn = True
                        break
                if is_worn:
                    continue  # 착용 중인 가방은 분실물 추적에서 제외

            filtered.append(item)
        return filtered

    def update(
        self,
        persons: List[dict],
        raw_items: List[dict],
        frame_idx: int
    ):
        persons_dict = {p['id']: p['bbox'] for p in persons}

        # 1. 사람이 착용 중인 물품 필터링
        items = self.filter_worn_items(raw_items, persons)

        # 2. 이번 프레임에서 매칭된 추적 객체 ID 집합
        matched_item_ids = set()

        for item_data in items:
            c_name = item_data['class_name']
            bbox = item_data['bbox']

            matched_id = None
            min_dist = float('inf')
            for tid, tracked in self.tracked_items.items():
                if tracked.class_name == c_name:
                    d = self._distance(tracked.bbox, bbox)
                    if d < 100.0 and d < min_dist:
                        min_dist = d
                        matched_id = tid

            if matched_id is not None:
                t = self.tracked_items[matched_id]
                t.bbox = [
                    0.7 * t.bbox[0] + 0.3 * bbox[0],
                    0.7 * t.bbox[1] + 0.3 * bbox[1],
                    0.7 * t.bbox[2] + 0.3 * bbox[2],
                    0.7 * t.bbox[3] + 0.3 * bbox[3],
                ]
                t.last_seen_frame = frame_idx
                t.missed_frames = 0
                matched_item_ids.add(matched_id)
            else:
                # 신규 정적 소지품 등록
                new_id = self.next_item_id
                self.next_item_id += 1

                closest_pid = None
                closest_pdist = float('inf')
                for p in persons:
                    d = self._distance(bbox, p['bbox'])
                    if d < closest_pdist:
                        closest_pdist = d
                        closest_pid = p['id']

                # 영상 초반 2초 동안은 주변 학생을 소유자로 매핑하고 SAFE로 시작
                owner_id = closest_pid
                initial_state = ItemState.SAFE

                new_tracked = TrackedItem(
                    item_id=new_id,
                    class_name=c_name,
                    bbox=bbox,
                    owner_id=owner_id,
                    initial_state=initial_state
                )
                new_tracked.last_seen_frame = frame_idx
                new_tracked.missed_frames = 0
                self.tracked_items[new_id] = new_tracked
                matched_item_ids.add(new_id)
                print(f"📦 [ITEM DETECT] {c_name} #{new_id} 등록 (소유자: Person #{owner_id})")

        # 3. 미검출된 객체 처리 (사람이 메고 나갔거나 화면에서 사라진 객체는 제거)
        for tid, tracked in list(self.tracked_items.items()):
            if tid not in matched_item_ids:
                tracked.missed_frames += 1
                # 사람이 메고 나가서 화면에서 사라진 가방류는 즉시 추적 해제 (유령 박스 방지)
                if tracked.class_name in {"backpack", "suitcase", "handbag"} and tracked.missed_frames > 4:
                    print(f"🗑️ [REMOVED] 퇴장한 물품 #{tid} ({tracked.class_name}) 추적 해제")
                    del self.tracked_items[tid]
                elif tracked.missed_frames > self.max_missed_frames:
                    del self.tracked_items[tid]

        # 4. FSM 상태 전이 (SAFE -> WARNING -> LOST)
        for item_id, tracked in list(self.tracked_items.items()):
            owner_nearby = False

            # 영상 초반 2초(48프레임)에는 주변에 사람이 있으므로 SAFE 유지
            if frame_idx < 48 and len(persons) > 0:
                owner_nearby = True
            elif tracked.owner_id is not None and tracked.owner_id in persons_dict:
                owner_bbox = persons_dict[tracked.owner_id]
                dist = self._distance(tracked.bbox, owner_bbox)
                if dist <= self.distance_threshold:
                    owner_nearby = True

            if tracked.state == ItemState.SAFE:
                if not owner_nearby:
                    tracked.state = ItemState.WARNING
                    tracked.warning_start_frame = frame_idx
                    print(f"🚨 [FSM F:{frame_idx}] {tracked.class_name} #{item_id}: SAFE -> WARNING (거리 이탈, 5초 타이머 시작!)")

            elif tracked.state == ItemState.WARNING:
                if owner_nearby:
                    tracked.state = ItemState.SAFE
                    tracked.warning_start_frame = None
                    print(f"✅ [FSM F:{frame_idx}] {tracked.class_name} #{item_id}: WARNING -> SAFE (소유자 복귀)")
                else:
                    elapsed = frame_idx - (tracked.warning_start_frame or frame_idx)
                    if elapsed >= self.lost_frame_threshold:
                        tracked.state = ItemState.LOST
                        print(f"🚨🚨 [FSM F:{frame_idx}] {tracked.class_name} #{item_id}: WARNING -> LOST (유예 만료! 분실물 확정 경보)")

            elif tracked.state == ItemState.LOST:
                pass


def draw_hud_and_boxes(
    frame: np.ndarray,
    tracker: LostItemTrackerFSMFixed,
    persons: List[dict],
    frame_idx: int,
    total_frames: int,
    fps: float
) -> np.ndarray:
    img_h, img_w = frame.shape[:2]
    annotated = frame.copy()

    # 1. 사람 바운딩 박스 (노란색)
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

    # 2. 소지품 FSM 상태 박스 및 라벨 (실제로 현재 검출되고 있는 객체만 렌더링!)
    for item_id, tracked in tracker.tracked_items.items():
        # 미검출된 상태의 객체는 렌더링하지 않음 (유령 박스 방지)
        if tracked.missed_frames > 2:
            continue

        x1, y1, x2, y2 = map(int, tracked.bbox)

        if tracked.state == ItemState.SAFE:
            color = (0, 255, 0)
            status_text = "SAFE"
            sub_text = "Owner Present"
        elif tracked.state == ItemState.WARNING:
            color = (0, 165, 255)
            status_text = "WARNING"
            elapsed_frames = frame_idx - (tracked.warning_start_frame or frame_idx)
            remaining_sec = max(0.0, (tracker.lost_frame_threshold - elapsed_frames) / fps)
            sub_text = f"Timer: {remaining_sec:.1f}s remaining"
        else:
            color = (0, 0, 255)
            status_text = "LOST ITEM ALERT!"
            sub_text = "Owner Abandoned"

        thickness = 4 if tracked.state == ItemState.LOST else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)

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
        cv2.putText(annotated, label_top, (x1 + 6, box_y1 + th1 + 4), font, font_scale, text_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, sub_text, (x1 + 6, box_y1 + th1 + th2 + 10), font, 0.5, text_color, 1, cv2.LINE_AA)

    # 3. 최상단 CCTV 관제 HUD
    hud_h = 60
    overlay = annotated[:hud_h, :].copy()
    cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.25, annotated[:hud_h, :], 0.75, 0, annotated[:hud_h, :])

    current_time_sec = frame_idx / fps
    total_time_sec = total_frames / fps

    # 활성 객체만 통계 집계
    active_items = [t for t in tracker.tracked_items.values() if t.missed_frames <= 2]
    safe_cnt = sum(1 for t in active_items if t.state == ItemState.SAFE)
    warn_cnt = sum(1 for t in active_items if t.state == ItemState.WARNING)
    lost_cnt = sum(1 for t in active_items if t.state == ItemState.LOST)

    hud_title = f"AI CCTV LAB MONITORING SYSTEM | CAM-01 [LECTURE ROOM 202]"
    cv2.putText(annotated, hud_title, (24, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    time_str = f"TIME: {current_time_sec:4.1f}s / {total_time_sec:4.1f}s ({frame_idx}/{total_frames}F) | DEMO_MODE (5s Grace Time)"
    cv2.putText(annotated, time_str, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    stat_x = img_w - 480
    cv2.rectangle(annotated, (stat_x, 12), (stat_x + 130, 48), (0, 140, 0), -1)
    cv2.putText(annotated, f"SAFE: {safe_cnt}", (stat_x + 16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(annotated, (stat_x + 145, 12), (stat_x + 285, 48), (0, 120, 230), -1)
    cv2.putText(annotated, f"WARN: {warn_cnt}", (stat_x + 158, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    lost_badge_color = (0, 0, 220) if lost_cnt > 0 else (60, 60, 60)
    cv2.rectangle(annotated, (stat_x + 300, 12), (stat_x + 455, 48), lost_badge_color, -1)
    cv2.putText(annotated, f"LOST: {lost_cnt}", (stat_x + 318, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def process_video_v3():
    input_path = Path("input_videos/Hailuo_Video_Use the uploaded image as the _570050405104934915.mp4")
    output_path = Path("output_results/lost_item_fsm_demo_v3_fixed.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ 입력 영상을 찾을 수 없습니다: {input_path}")
        return

    print("=" * 70)
    print("🚀 [분실물 감지 파이프라인 V3 - 착용 물품 및 유령 박스 제거 버전]")
    print(f"   입력 영상: {input_path.name}")
    print(f"   출력 영상: {output_path}")
    print("=" * 70)

    print("📦 [1/4] YOLOv8x 모델 로딩...")
    model = YOLO("yolov8x.pt")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 [2/4] 비디오 스펙: {width}x{height} @ {fps:.1f} FPS (총 {total_frames} 프레임)")

    frame_stride = 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    tracker = LostItemTrackerFSMFixed(
        fps=fps,
        distance_threshold=400.0,
        lost_time_seconds=5.0,
        max_missed_frames=6
    )

    print(f"⚙️ [3/4] 렌더링 시작 (Stride={frame_stride})...")
    start_time = time.time()

    last_persons = []
    last_items = []
    frame_idx = 0

    # 추적 대상: 실제 귀중품 및 소지품
    TARGET_ITEM_CLASSES = {"laptop", "mouse", "cell phone"}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % frame_stride == 1 or frame_idx == 1:
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

            tracker.update(last_persons, last_items, frame_idx)
        else:
            tracker.update(last_persons, last_items, frame_idx)

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
                f"속도: {proc_fps:4.1f} FPS | 소요: {elapsed:5.1f}초",
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
    process_video_v3()
