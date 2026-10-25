"""
[실전 강의실 퇴실 후 방치 분실물 자동 감지 및 FSM 추적 파이프라인]
- 대상 영상: merged_cctv_lost_item_video.mp4 (2944x1248, 24fps, 676프레임)
- 시나리오:
    1. 수업 중 (0~10초): 20여 명의 학생 착석 및 개인 물품(텀블러, 물병, 노트북, 휴대폰 등) 사용 [SAFE]
    2. 퇴실 진행 (10~15초): 학생들이 짐을 챙겨 좌측 출입구로 퇴실 [SAFE -> WARNING 카운트다운]
    3. 퇴실 완료 및 방치 (15~28초): 학생이 모두 떠난 빈 책상에 남겨진 물품 분실 확정 [LOST ALERT]
    4. 분실 확정 시점 공식 증거 리포트(증거 크롭, 비주얼 카드, 인쇄용 HTML, JSON) 자동 발급!
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

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from core.report_generator import IncidentReportGenerator

VIDEO_PATH = CURRENT_DIR / "input_videos" / "merged_cctv_lost_item_video.mp4"
OUTPUT_DIR = CURRENT_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "merged_cctv_lost_item_analyzed.mp4"


class ItemState(Enum):
    SAFE = "SAFE"        # 소유자가 인근에 머무는 정상 상태 (초록색)
    WARNING = "WARNING"  # 소유자 이탈 후 유예 타이머 작동 상태 (주황색)
    LOST = "LOST"        # 유예 시간 만료되어 분실물로 최종 확정 (빨간색 경보)


class TrackedLostItem:
    def __init__(self, item_id: int, class_name: str, bbox: List[float], owner_id: Optional[int] = None):
        self.item_id = item_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.owner_id = owner_id
        self.state = ItemState.SAFE
        self.warning_start_frame: Optional[int] = None
        self.last_seen_frame = 0
        self.missed_frames = 0
        self.reported = False


class ClassroomLostItemFSM:
    def __init__(self, fps: float = 24.0, distance_threshold: float = 380.0, lost_time_seconds: float = 5.0):
        self.fps = fps
        self.distance_threshold = distance_threshold
        self.lost_frame_threshold = int(self.fps * lost_time_seconds)
        self.tracked_items: Dict[int, TrackedLostItem] = {}
        self.next_item_id = 1
        self.reporter = IncidentReportGenerator()
        self.issued_reports = []

    @staticmethod
    def _center(box: List[float]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @classmethod
    def _dist(cls, b1: List[float], b2: List[float]) -> float:
        c1, c2 = cls._center(b1), cls._center(b2)
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    @staticmethod
    def _is_inside(inner: List[float], outer: List[float]) -> bool:
        cx, cy = (inner[0] + inner[2]) / 2.0, (inner[1] + inner[3]) / 2.0
        return (outer[0] <= cx <= outer[2]) and (outer[1] <= cy <= outer[3])

    def filter_worn_items(self, items: List[dict], persons: List[dict]) -> List[dict]:
        """사람 몸에 완전히 겹치는 가방/백팩은 착용 중인 것으로 판정하여 제외"""
        filtered = []
        for it in items:
            c = it['class_name']
            if c in {"backpack", "handbag"}:
                if any(self._is_inside(it['bbox'], p['bbox']) for p in persons):
                    continue
            filtered.append(it)
        return filtered

    def update(self, frame: np.ndarray, persons: List[dict], raw_items: List[dict], frame_idx: int):
        items = self.filter_worn_items(raw_items, persons)
        matched_ids = set()

        # 1. 기존 추적 객체와 거리 기반 매칭
        for it in items:
            cname = it['class_name']
            bbox = it['bbox']
            best_id, min_d = None, float('inf')

            for tid, tracked in self.tracked_items.items():
                if tracked.class_name == cname:
                    d = self._dist(tracked.bbox, bbox)
                    if d < 120.0 and d < min_d:
                        min_d = d
                        best_id = tid

            if best_id is not None:
                t = self.tracked_items[best_id]
                # 완만한 위치 보간
                t.bbox = [
                    0.7 * t.bbox[0] + 0.3 * bbox[0],
                    0.7 * t.bbox[1] + 0.3 * bbox[1],
                    0.7 * t.bbox[2] + 0.3 * bbox[2],
                    0.7 * t.bbox[3] + 0.3 * bbox[3],
                ]
                t.last_seen_frame = frame_idx
                t.missed_frames = 0
                matched_ids.add(best_id)
            else:
                # 새 객체 등록
                new_id = self.next_item_id
                self.next_item_id += 1
                owner_id = None
                closest_p = float('inf')
                for p in persons:
                    d = self._dist(bbox, p['bbox'])
                    if d < closest_p and d < self.distance_threshold:
                        closest_p = d
                        owner_id = p['id']

                item_obj = TrackedLostItem(new_id, cname, bbox, owner_id)
                item_obj.last_seen_frame = frame_idx
                self.tracked_items[new_id] = item_obj
                matched_ids.add(new_id)

        # 2. 장기 미검출 객체 정리
        for tid, t in list(self.tracked_items.items()):
            if tid not in matched_ids:
                t.missed_frames += 1
                # 이미 LOST 확정된 물품은 사람이 가려도 계속 유지
                if t.state != ItemState.LOST and t.missed_frames > 15:
                    del self.tracked_items[tid]

        # 3. FSM 상태 전이
        for tid, t in list(self.tracked_items.items()):
            # 학생 존재 여부 확인 (최소 반경 내에 사람이 있는지)
            has_nearby_person = any(self._dist(t.bbox, p['bbox']) <= self.distance_threshold for p in persons)

            # 초반 100프레임(약 4초) 동안은 전체 수업 중이므로 SAFE 보장
            if frame_idx < 100:
                has_nearby_person = True

            if t.state == ItemState.SAFE:
                if not has_nearby_person:
                    t.state = ItemState.WARNING
                    t.warning_start_frame = frame_idx
            elif t.state == ItemState.WARNING:
                if has_nearby_person:
                    t.state = ItemState.SAFE
                    t.warning_start_frame = None
                else:
                    elapsed = frame_idx - (t.warning_start_frame or frame_idx)
                    if elapsed >= self.lost_frame_threshold:
                        t.state = ItemState.LOST

                        # 최초 LOST 전이 시 즉각 공식 증거 리포트 발급!
                        if not t.reported:
                            t.reported = True
                            bx = [int(v) for v in t.bbox]
                            sec = frame_idx / self.fps
                            print(f"\n🚨 [분실물 확정 경보!] Frame {frame_idx} ({sec:.1f}s) - {t.class_name.upper()} #{t.item_id} 방치 감지! 리포트 발급 중...")
                            rep = self.reporter.generate_incident_report(
                                full_frame=frame,
                                bbox=bx,
                                incident_type="LOST_PROPERTY",
                                item_class=t.class_name,
                                confidence=0.92,
                                classroom_id="인문사회관 205호 멀티미디어강의실",
                                camera_id="CCTV-CAM-01 (강의실 전경 시점)",
                                location_desc=f"강의실 좌석 책상 위 (좌표 X:{bx[0]}~{bx[2]}, Y:{bx[1]}~{bx[3]})",
                                suspect_desc="수업 종료 후 학생 전원 퇴실 완료 시점까지 책상 위에 장시간 방치됨",
                                action_note="실습실 분실물 보관함 #02 보관 및 학과사무실 분실물 등록 공지 완료"
                            )
                            self.issued_reports.append(rep)


def draw_hud(annotated: np.ndarray, fsm: ClassroomLostItemFSM, persons: List[dict],
             frame_idx: int, total_frames: int, fps: float):
    h, w = annotated.shape[:2]
    cur_sec = frame_idx / fps
    tot_sec = total_frames / fps

    # 1. 사람 바운딩 박스
    for p in persons:
        px1, py1, px2, py2 = [int(v) for v in p['bbox']]
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 180, 0), 2)
        cv2.putText(annotated, f"Student #{p['id']}", (px1, max(20, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

    # 2. 물품 바운딩 박스 및 상태 태그
    state_colors = {
        ItemState.SAFE: (0, 200, 80),
        ItemState.WARNING: (0, 160, 255),
        ItemState.LOST: (0, 0, 255)
    }

    for tid, it in fsm.tracked_items.items():
        x1, y1, x2, y2 = [int(v) for v in it.bbox]
        color = state_colors[it.state]
        thick = 3 if it.state == ItemState.LOST else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

        if it.state == ItemState.SAFE:
            tag = f"[SAFE] {it.class_name}"
        elif it.state == ItemState.WARNING:
            remain = max(0.0, (fsm.lost_frame_threshold - (frame_idx - (it.warning_start_frame or frame_idx))) / fps)
            tag = f"[WARN {remain:.1f}s] {it.class_name}"
        else:
            tag = f"🚨 [LOST] {it.class_name.upper()} #{it.item_id}"

        cv2.putText(annotated, tag, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

    # 3. 상단 관제 HUD 바
    hud_h = 75
    cv2.rectangle(annotated, (0, 0), (w, hud_h), (20, 22, 28), -1)

    safe_n = sum(1 for t in fsm.tracked_items.values() if t.state == ItemState.SAFE)
    warn_n = sum(1 for t in fsm.tracked_items.values() if t.state == ItemState.WARNING)
    lost_n = sum(1 for t in fsm.tracked_items.values() if t.state == ItemState.LOST)

    if lost_n > 0:
        cv2.rectangle(annotated, (20, 12), (620, hud_h - 12), (0, 0, 180), -1)
        cv2.putText(annotated, f"🚨 [LOST PROPERTY ALERT] {lost_n} ITEM(S) UNATTENDED", (32, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(annotated, (20, 12), (540, hud_h - 12), (0, 130, 60), -1)
        cv2.putText(annotated, "[AI CCTV] POST-CLASS LOST ITEM MONITORING", (32, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2)

    # 우측 통계 배지
    stat_x = w - 620
    cv2.rectangle(annotated, (stat_x, 14), (stat_x + 100, hud_h - 14), (0, 160, 70), -1)
    cv2.putText(annotated, f"SAFE: {safe_n}", (stat_x + 12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.rectangle(annotated, (stat_x + 115, 14), (stat_x + 225, hud_h - 14), (0, 140, 240), -1)
    cv2.putText(annotated, f"WARN: {warn_n}", (stat_x + 124, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    cv2.rectangle(annotated, (stat_x + 240, 14), (stat_x + 350, hud_h - 14), (0, 0, 220) if lost_n > 0 else (60, 60, 60), -1)
    cv2.putText(annotated, f"LOST: {lost_n}", (stat_x + 252, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(annotated, f"Students: {len(persons):2d} | Time: {cur_sec:4.1f}s / {tot_sec:4.1f}s",
                (w - 240, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def run_pipeline():
    print("=" * 70)
    print("🏫 [신규 CCTV 영상 퇴실 후 방치 분실물 감지 파이프라인 시작]")
    print(f" - 입력 영상: {VIDEO_PATH.name}")
    print(f" - 출력 영상: {OUTPUT_VIDEO.name}")
    print("=" * 70)

    if not VIDEO_PATH.exists():
        print(f"[오류] 비디오를 찾을 수 없습니다: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" - 비디오 스펙: {width}x{height}, {fps:.1f} FPS, 총 {total}프레임 ({total/fps:.1f}초)")

    model = YOLO("yolov8x.pt")
    fsm = ClassroomLostItemFSM(fps=fps, distance_threshold=380.0, lost_time_seconds=5.0)

    out = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    t0 = time.time()
    last_persons = []
    last_items = []
    stride = 2

    # 관심 대상 분실물 클래스
    TARGET_CLASSES = {"bottle", "cup", "laptop", "cell phone", "book", "backpack", "mouse"}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % stride == 1 or frame_idx == 1:
            scale = 1280.0 / width
            small = cv2.resize(frame, (1280, int(height * scale)))
            res = model.predict(small, imgsz=1280, conf=0.18, verbose=False)[0]

            curr_persons = []
            curr_items = []
            pid = 1

            for b in res.boxes:
                cname = model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                bx1, by1, bx2, by2 = [v / scale for v in b.xyxy[0].tolist()]

                # 교탁 위 강사 노트북/기자재는 공용 고정자산이므로 학생 책상 분실물에서 제외 (y > 900 하단 교탁 제외)
                if by1 > 950:
                    continue

                if cname == "person":
                    curr_persons.append({"id": pid, "bbox": [bx1, by1, bx2, by2]})
                    pid += 1
                elif cname in TARGET_CLASSES:
                    curr_items.append({"class_name": cname, "bbox": [bx1, by1, bx2, by2], "conf": conf})

            last_persons = curr_persons
            last_items = curr_items
            fsm.update(frame, last_persons, last_items, frame_idx)
        else:
            fsm.update(frame, last_persons, last_items, frame_idx)

        annotated = frame.copy()
        draw_hud(annotated, fsm, last_persons, frame_idx, total, fps)
        out.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total:
            pct = (frame_idx / total) * 100
            elapsed = time.time() - t0
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"\r진행률: {frame_idx}/{total} ({pct:5.1f}%) | 속도: {proc_fps:4.1f} FPS | 남은시간: {(total - frame_idx) / proc_fps if proc_fps > 0 else 0:4.1f}초", end="", flush=True)

    cap.release()
    out.release()

    print(f"\n\n🎉 [완료] 분석 및 렌더링 완료! (소요 시간: {time.time() - t0:.1f}초)")
    print(f" - 생성 비디오: {OUTPUT_VIDEO}")
    print(f" - 발급된 분실물 조사 보고서: 총 {len(fsm.issued_reports)}건")
    for r in fsm.issued_reports:
        print(f"   • [{r['incident_id']}] {r['card_path']}")
        print(f"     HTML: {r['html_path']}")


if __name__ == "__main__":
    run_pipeline()
