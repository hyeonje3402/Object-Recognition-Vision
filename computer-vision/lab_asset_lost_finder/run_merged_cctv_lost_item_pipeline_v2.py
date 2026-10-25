"""
[강의실 방치/반출 물품 추적 파이프라인 v2]
- 대상 영상: merged_cctv_lost_item_video.mp4
- v1 대비 개선점
    1. SAHI 슬라이싱 추론(원본 해상도)으로 책상 위 소형 물품 검출률 향상
    2. 검출 누락(미탐)과 실제 사라짐을 템플릿 매칭으로 구분 -> 남아있는 물건을 끝까지 추적
    3. 물건이 사라진 경우 TAKEN(가져감) 상태로 판정
       - 퇴실 완료 이후 사라지면 TAKEN_SUSPECT (제3자 반출 의심, 증거 리포트 발급)
    4. 프레임 번호 하드코딩 제거: 인원수 변화(피크 대비 감소)로 퇴실 여부를 자동 판단
사용법: python run_merged_cctv_lost_item_pipeline_v2.py [--frames N]
"""
import argparse
import pickle
import sys
import time
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from core.report_generator import IncidentReportGenerator

VIDEO_PATH = CURRENT_DIR / "input_videos" / "merged_cctv_lost_item_video.mp4"
OUTPUT_DIR = CURRENT_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "merged_cctv_lost_item_analyzed_v2.mp4"

TARGET_CLASSES = {"bottle", "cup", "laptop", "cell phone", "book", "backpack", "handbag", "mouse"}
PODIUM_Y = 950           # 교탁(고정 공용 기자재) 영역 하한선(px) - 학생 소지품에서 제외
DETECT_STRIDE = 6        # N프레임마다 SAHI 검출
NEAR_PX = 200            # 물건 중심 ~ 사람 박스 사이 최소거리가 이 값 이하면 '사람이 곁에 있음'
MATCH_PX = 110           # 같은 물건으로 볼 중심 거리
TEMPLATE_OK = 0.55       # 템플릿 매칭 점수 이상이면 물건이 아직 그 자리에 있음
CONFIRM_HITS = 3         # 같은 자리에서 이 횟수 이상 연속 검출(정지 확인)되어야 정식 물품으로 등록
ANCHOR_MOVE_PX = 60      # 미확정 물품이 최초 위치에서 이만큼 움직이면 '이동 중'으로 보고 재시작
MAX_ITEM_W, MAX_ITEM_H = 300, 250   # 이보다 큰 박스는 물품이 아닌 오검출로 제외
MIN_STATIC_SECONDS = 3.0 # 이 시간 이상 정지해 있던 물품만 TAKEN 판정 대상
CACHE_PATH = OUTPUT_DIR / "v2_detections.pkl"


class ItemState(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    LOST = "LOST"
    TAKEN = "TAKEN"                  # 퇴실 전 소유자 본인이 챙겨감(정상)
    TAKEN_SUSPECT = "TAKEN_SUSPECT"  # 퇴실 완료 후 사라짐(반출 의심)


STATE_COLORS = {
    ItemState.SAFE: (0, 200, 80),
    ItemState.WARNING: (0, 160, 255),
    ItemState.LOST: (0, 0, 255),
    ItemState.TAKEN: (200, 200, 200),
    ItemState.TAKEN_SUSPECT: (255, 0, 255),
}


class Item:
    def __init__(self, item_id: int, cname: str, bbox: List[float], frame: np.ndarray, frame_idx: int):
        self.id = item_id
        self.cname = cname
        self.bbox = list(bbox)
        self.anchor = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.first_seen = frame_idx
        self.hits = 1
        self.state = ItemState.SAFE
        self.last_det = frame_idx
        self.last_present = frame_idx
        self.last_near: Optional[int] = None
        self.warn_start: Optional[int] = None
        self.missing_since: Optional[int] = None
        self.event_frame: Optional[int] = None
        self.reported = False
        self.template = self._crop(frame)

    def center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def _crop(self, frame):
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    def refresh_template(self, frame):
        t = self._crop(frame)
        if t.shape[0] >= 8 and t.shape[1] >= 8:
            self.template = t


def rect_dist(px: float, py: float, box: List[float]) -> float:
    dx = max(box[0] - px, 0, px - box[2])
    dy = max(box[1] - py, 0, py - box[3])
    return float(np.hypot(dx, dy))


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    def __init__(self, fps: float, lost_seconds: float = 5.0, take_seconds: float = 1.0):
        self.fps = fps
        self.lost_frames = int(fps * lost_seconds)
        self.take_frames = int(fps * take_seconds)
        self.items: Dict[int, Item] = OrderedDict()
        self.next_id = 1
        self.peak_persons = 0
        self.frame_buffer: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.reporter = IncidentReportGenerator()
        self.reports: List[dict] = []

    # ---- 퇴실 자동 판단 ----
    def dismissed(self, n_persons: int) -> bool:
        return self.peak_persons >= 5 and n_persons <= max(2, int(0.2 * self.peak_persons))

    def _near_person(self, item: Item, persons: List[dict]) -> bool:
        cx, cy = item.center()
        return any(rect_dist(cx, cy, p["bbox"]) <= NEAR_PX for p in persons)

    def _occluded(self, item: Item, persons: List[dict]) -> bool:
        cx, cy = item.center()
        return any(p["bbox"][0] - 30 <= cx <= p["bbox"][2] + 30 and p["bbox"][1] - 30 <= cy <= p["bbox"][3] + 30
                   for p in persons)

    def remember_frame(self, frame, frame_idx):
        self.frame_buffer[frame_idx] = frame.copy()
        while len(self.frame_buffer) > 16:
            self.frame_buffer.popitem(last=False)

    def _template_present(self, item: Item, frame) -> bool:
        x1, y1, x2, y2 = [int(v) for v in item.bbox]
        pad = 25
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        region = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        t = item.template
        if t is None or t.shape[0] < 8 or t.shape[1] < 8 or region.shape[0] < t.shape[0] or region.shape[1] < t.shape[1]:
            return True  # 판단 불가 시 존재로 간주(오탐 방지)
        score = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED).max()
        return score >= TEMPLATE_OK

    # ---- 검출 프레임 갱신 ----
    def on_detection(self, frame, persons, dets, frame_idx):
        self.peak_persons = max(self.peak_persons, len(persons))
        self.remember_frame(frame, frame_idx)
        live = [it for it in self.items.values() if it.state in (ItemState.SAFE, ItemState.WARNING, ItemState.LOST)
                or it.hits < CONFIRM_HITS]
        matched = set()

        for d in dets:
            if any(self._is_worn(d, p) for p in persons) and d["cname"] in {"backpack", "handbag"}:
                continue
            best, bd = None, 1e9
            for it in live:
                if it.id in matched:
                    continue
                cx, cy = it.center()
                dcx, dcy = (d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2
                dist = float(np.hypot(cx - dcx, cy - dcy))
                same = it.cname == d["cname"]
                if (same and dist < MATCH_PX and dist < bd) or (not same and iou(it.bbox, d["bbox"]) > 0.4 and dist < bd):
                    best, bd = it, dist
            if best is not None:
                b = best.bbox
                best.bbox = [0.6 * b[i] + 0.4 * d["bbox"][i] for i in range(4)]
                if best.hits < CONFIRM_HITS:
                    ncx, ncy = best.center()
                    if np.hypot(ncx - best.anchor[0], ncy - best.anchor[1]) > ANCHOR_MOVE_PX:
                        best.anchor, best.hits, best.first_seen = (ncx, ncy), 0, frame_idx  # 이동 중 -> 재시작
                best.hits += 1
                best.last_det = best.last_present = frame_idx
                best.missing_since = None
                matched.add(best.id)
                if best.state == ItemState.SAFE:
                    best.refresh_template(frame)
            else:
                it = Item(self.next_id, d["cname"], d["bbox"], frame, frame_idx)
                self.items[it.id] = it
                self.next_id += 1
                matched.add(it.id)

        # 미검출 항목: 템플릿 매칭으로 '미탐'과 '실제 사라짐' 구분
        for it in list(self.items.values()):
            if it.id in matched or it.state in (ItemState.TAKEN, ItemState.TAKEN_SUSPECT):
                continue
            if it.hits < CONFIRM_HITS:
                if frame_idx - it.last_det > DETECT_STRIDE * 3:
                    del self.items[it.id]
                continue
            if self._template_present(it, frame):
                it.last_present = frame_idx
                it.missing_since = None
            elif self._occluded(it, persons):
                pass  # 사람에게 가려진 상태 - 판정 보류
            else:
                if it.missing_since is None:
                    it.missing_since = frame_idx
                recently_attended = it.last_near is not None and it.last_near >= it.missing_since - int(3 * self.fps)
                was_static = (it.last_present - it.first_seen) >= MIN_STATIC_SECONDS * self.fps
                if frame_idx - it.missing_since >= self.take_frames and recently_attended and was_static:
                    self._mark_taken(it, frame, persons, frame_idx)

    @staticmethod
    def _is_worn(d, p) -> bool:
        cx, cy = (d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2
        b = p["bbox"]
        return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]

    def _mark_taken(self, it: Item, frame, persons, frame_idx):
        it.event_frame = frame_idx
        if self.dismissed(len(persons)):
            it.state = ItemState.TAKEN_SUSPECT
            self._report(it, frame, "ASSET_THEFT", frame_idx,
                         "퇴실 완료 후 책상 위 물품이 사라짐 - 제3자 반출 의심", "CCTV 영상 확인 및 학과사무실 도난 신고 접수")
        else:
            it.state = ItemState.TAKEN

    def _report(self, it: Item, frame, itype, frame_idx, suspect, note):
        if it.reported:
            return
        it.reported = True
        src = None
        for k in sorted(self.frame_buffer.keys(), reverse=True):
            if k <= (it.last_present or frame_idx):
                src = self.frame_buffer[k]
                break
        bx = [int(v) for v in it.bbox]
        print(f"\n[{itype}] frame {frame_idx} ({frame_idx / self.fps:.1f}s) {it.cname} #{it.id}")
        rep = self.reporter.generate_incident_report(
            full_frame=frame, bbox=bx, incident_type=itype, item_class=it.cname, confidence=0.9,
            classroom_id="인문사회관 205호 멀티미디어강의실", camera_id="CCTV-CAM-01 (강의실 전경 시점)",
            location_desc=f"강의실 좌석 책상 위 (좌표 X:{bx[0]}~{bx[2]}, Y:{bx[1]}~{bx[3]})",
            suspect_desc=suspect, action_note=note, crop_source_frame=src)
        self.reports.append(rep)

    # ---- 매 프레임 상태 전이 ----
    def on_frame(self, frame, persons, frame_idx):
        dismissed = self.dismissed(len(persons))
        for it in self.items.values():
            if it.hits < CONFIRM_HITS or it.state in (ItemState.TAKEN, ItemState.TAKEN_SUSPECT):
                continue
            near = self._near_person(it, persons)
            if near:
                it.last_near = frame_idx
            if it.state == ItemState.SAFE:
                if not near and (it.last_near is not None or dismissed):
                    it.state, it.warn_start = ItemState.WARNING, frame_idx
            elif it.state == ItemState.WARNING:
                if near:
                    it.state, it.warn_start = ItemState.SAFE, None
                elif frame_idx - (it.warn_start or frame_idx) >= self.lost_frames and dismissed:
                    it.state = ItemState.LOST
                    self._report(it, frame, "LOST_PROPERTY", frame_idx,
                                 "수업 종료 후 학생 대부분 퇴실 완료 시점까지 책상 위에 방치됨",
                                 "실습실 분실물 보관함 보관 및 학과사무실 분실물 등록 공지")
        # 사라진 TAKEN 고스트 정리(너무 오래된 표시 제거는 draw에서 처리)


def draw(frame, tr: Tracker, persons, frame_idx, total, fps):
    out = frame.copy()
    h, w = out.shape[:2]
    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
    counts = {s: 0 for s in ItemState}
    for it in tr.items.values():
        if it.hits < CONFIRM_HITS:
            continue
        counts[it.state] += 1
        if it.state in (ItemState.TAKEN, ItemState.TAKEN_SUSPECT) and frame_idx - (it.event_frame or 0) > fps * 6:
            continue  # 가져간 물건 표시는 6초간 유지
        x1, y1, x2, y2 = [int(v) for v in it.bbox]
        col = STATE_COLORS[it.state]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if it.state != ItemState.SAFE else 2)
        if it.state == ItemState.WARNING:
            rem = max(0.0, (tr.lost_frames - (frame_idx - (it.warn_start or frame_idx))) / fps)
            tag = f"WARN {rem:.1f}s {it.cname}"
        elif it.state == ItemState.TAKEN_SUSPECT:
            tag = f"TAKEN? {it.cname} #{it.id}"
        else:
            tag = f"{it.state.value} {it.cname}"
        cv2.putText(out, tag, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.rectangle(out, (0, 0), (w, 75), (20, 22, 28), -1)
    alert = counts[ItemState.LOST] + counts[ItemState.TAKEN_SUSPECT]
    cv2.rectangle(out, (20, 12), (620, 63), (0, 0, 180) if alert else (0, 130, 60), -1)
    msg = f"ALERT: LOST {counts[ItemState.LOST]} / TAKEN? {counts[ItemState.TAKEN_SUSPECT]}" if alert else "AI CCTV ITEM MONITORING"
    cv2.putText(out, msg, (32, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    stats = (f"SAFE {counts[ItemState.SAFE]} | WARN {counts[ItemState.WARNING]} | LOST {counts[ItemState.LOST]} | "
             f"PICKED {counts[ItemState.TAKEN]} | TAKEN? {counts[ItemState.TAKEN_SUSPECT]}")
    cv2.putText(out, stats, (w - 1150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, f"Persons {len(persons)} (peak {tr.peak_persons}) | {frame_idx / fps:4.1f}s / {total / fps:4.1f}s",
                (w - 1150, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return out


def detect(model, frame):
    res = get_sliced_prediction(frame, model, slice_height=700, slice_width=1100,
                                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                perform_standard_pred=True, postprocess_type="NMS",
                                postprocess_match_threshold=0.5, verbose=0)
    persons, dets = [], []
    for o in res.object_prediction_list:
        name = o.category.name
        b = o.bbox.to_xyxy()
        if b[1] > PODIUM_Y:
            continue
        if name == "person":
            persons.append({"id": len(persons) + 1, "bbox": b})
        elif name in TARGET_CLASSES:
            dets.append({"cname": name, "bbox": b})
    return persons, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="처리할 최대 프레임 수(0=전체)")
    ap.add_argument("--redetect", action="store_true", help="검출 캐시 무시하고 SAHI 재실행")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frames:
        total = min(total, args.frames)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"input={VIDEO_PATH.name} {w}x{h} {fps:.1f}fps frames={total}")

    model = None
    cache = {}
    if CACHE_PATH.exists() and not args.redetect:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"detection cache loaded: {len(cache)} frames")
    else:
        model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path="yolov8x.pt",
                                                   confidence_threshold=0.15, device="cpu")
    tr = Tracker(fps)
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    persons: List[dict] = []
    t0 = time.time()
    idx = 0
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx == 1 or idx % DETECT_STRIDE == 0:
            if idx in cache:
                persons, raw = cache[idx]
            else:
                persons, raw = detect(model, frame)
                cache[idx] = (persons, raw)
            dets = [d for d in raw if (d["bbox"][2] - d["bbox"][0]) <= MAX_ITEM_W and (d["bbox"][3] - d["bbox"][1]) <= MAX_ITEM_H]
            tr.on_detection(frame, persons, dets, idx)
        tr.on_frame(frame, persons, idx)
        out.write(draw(frame, tr, persons, idx, total, fps))
        if idx % 24 == 0:
            el = time.time() - t0
            print(f"{idx}/{total} ({idx / total * 100:.0f}%) eta {(total - idx) * el / idx:.0f}s "
                  f"persons={len(persons)} items={sum(1 for i in tr.items.values() if i.hits >= CONFIRM_HITS)}", flush=True)
    cap.release()
    out.release()
    if model is not None:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    final = {s.value: 0 for s in ItemState}
    for it in tr.items.values():
        if it.hits >= CONFIRM_HITS:
            final[it.state.value] += 1
    print("DONE", f"{time.time() - t0:.0f}s", "peak_persons=", tr.peak_persons, "final=", final)
    print("video:", OUTPUT_VIDEO)
    for r in tr.reports:
        print(" report:", r["incident_id"], r["card_path"])


if __name__ == "__main__":
    main()
