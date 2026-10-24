"""
[동적 상호작용 기반 도난 판정 엔진 - core/interaction_tracker.py]

프레임 번호 하드코딩 없이, 순수 시공간 조건만으로 도난을 판정한다.

  1) 접촉(Interaction)   : 추적 중인 사람(ByteTrack ID)이 등록 자산 근처에 접근
  2) 결손(Missing)       : 자산 위치에서 물품이 사라진 상태가 일정 시간 지속
                           (사람 몸/손에 가려진 구간은 결손 카운트에서 제외)
  3) 이탈(Departure)     : 마지막으로 접촉한 사람이 자산에서 멀어지며 출입문 방향으로 이동
  => 세 조건이 모두 만족될 때만 THEFT_ALERT 발령

시간 단위는 모두 '프레임 수 / fps' 로 환산하므로 영상 처리 속도와 무관하게 동작한다.
"""
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[float, float, float, float]


class TheftState(Enum):
    SAFE = "SAFE"                        # 정상 거치
    INTERACTING = "INTERACTING"          # 사람이 접촉/근접 중
    MISSING_SUSPECT = "MISSING_SUSPECT"  # 물품 소멸 확정, 접촉자의 이탈 경로 감시 중
    THEFT_ALERT = "THEFT_ALERT"          # 도난 확정


def box_center(b) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def point_to_box_distance(px: float, py: float, b) -> float:
    """점과 박스 사이의 최단 거리 (박스 내부면 0)"""
    dx = max(b[0] - px, 0.0, px - b[2])
    dy = max(b[1] - py, 0.0, py - b[3])
    return math.hypot(dx, dy)


class AssetPresenceChecker:
    """
    등록 기준 프레임의 패치와 현재 프레임을 템플릿 매칭으로 비교하여
    '해당 위치에 물품이 아직 있는가'를 프레임마다 판정한다.
    (소형 물품은 프레임 단위 YOLO 검출이 불안정하므로 기준 패치 매칭을 사용)
    """

    def __init__(self, baseline_frame: np.ndarray, assets: Dict[str, dict],
                 search_pad: int = 18, match_thresh: float = 0.5):
        self.search_pad = search_pad
        self.match_thresh = match_thresh
        self.templates: Dict[str, np.ndarray] = {}
        gray = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
        for aid, a in assets.items():
            x1, y1, x2, y2 = [int(v) for v in a["bbox"]]
            patch = gray[max(0, y1):y2, max(0, x1):x2]
            if patch.shape[0] >= 6 and patch.shape[1] >= 6:
                self.templates[aid] = patch.copy()

    def check(self, frame: np.ndarray, assets: Dict[str, dict]) -> Dict[str, float]:
        """asset_id -> 매칭 점수(0~1). 점수가 낮을수록 물품이 없음."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scores: Dict[str, float] = {}
        for aid, tpl in self.templates.items():
            x1, y1, x2, y2 = [int(v) for v in assets[aid]["bbox"]]
            sx1, sy1 = max(0, x1 - self.search_pad), max(0, y1 - self.search_pad)
            sx2, sy2 = min(w, x2 + self.search_pad), min(h, y2 + self.search_pad)
            win = gray[sy1:sy2, sx1:sx2]
            if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
                continue
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            scores[aid] = float(np.nan_to_num(res.max(), nan=0.0))
        return scores

    def presence(self, scores: Dict[str, float]) -> Dict[str, bool]:
        return {aid: s >= self.match_thresh for aid, s in scores.items()}


class _AssetRecord:
    def __init__(self):
        self.state = TheftState.SAFE
        self.missing_frames = 0
        self.interactor_id: Optional[int] = None
        self.interactor_frame = -10 ** 9
        self.interactor_door_dist: Optional[float] = None
        self.missing_since_frame: Optional[int] = None


class DynamicTheftDetector:
    def __init__(
        self,
        exit_door_bbox: Tuple[int, int, int, int],
        fps: float = 24.0,
        interact_dist: float = 120.0,
        occlusion_margin: float = 0.0,
        confirm_missing_sec: float = 0.6,
        interaction_memory_sec: float = 10.0,
        door_trigger_dist: float = 250.0,
        approach_px: float = 150.0,
        min_departure_px: float = 150.0,
        lost_track_sec: float = 2.0,
    ):
        """
        :param exit_door_bbox: 출입문 ROI [x1,y1,x2,y2] (픽셀)
        :param interact_dist: 자산 중심 ~ 사람 박스 거리가 이 값 이하면 '접촉'
        :param occlusion_margin: 사람 박스가 자산을 이 마진 이내로 덮으면 '가려짐'(결손 카운트 보류)
        :param confirm_missing_sec: 물품 소멸이 이 시간 이상 지속되어야 결손 확정
        :param interaction_memory_sec: 마지막 접촉 기록 유지 시간
        :param door_trigger_dist: 접촉자와 출입문 ROI 사이 거리가 이 값 이하면 '출입문 도달'
        :param approach_px: 접촉 시점 대비 출입문에 이만큼 가까워지면 '출입문 방향 이동'
        :param min_departure_px: 도난 확정에 필요한 자산 ~ 접촉자 최소 이격 거리
        :param lost_track_sec: 접촉자 트랙이 사라진 뒤에도 마지막 위치로 판정 가능한 시간
        """
        self.exit_door = exit_door_bbox
        self.fps = fps
        self.interact_dist = interact_dist
        self.occlusion_margin = occlusion_margin
        self.confirm_missing_frames = max(1, int(confirm_missing_sec * fps))
        self.memory_frames = int(interaction_memory_sec * fps)
        self.door_trigger_dist = door_trigger_dist
        self.approach_px = approach_px
        self.min_departure_px = min_departure_px
        self.lost_track_frames = int(lost_track_sec * fps)

        self.records: Dict[str, _AssetRecord] = {}
        self.person_last_seen: Dict[int, Tuple[List[float], int]] = {}

    def _door_dist(self, bbox) -> float:
        cx, cy = box_center(bbox)
        return point_to_box_distance(cx, cy, self.exit_door)

    def get_state(self, asset_id: str) -> TheftState:
        rec = self.records.get(asset_id)
        return rec.state if rec else TheftState.SAFE

    def update(
        self,
        frame_idx: int,
        registered_assets: Dict[str, dict],   # {id: {"class", "bbox", ...}}
        presence: Dict[str, bool],            # {id: 물품이 현재 위치에 있는가}
        tracked_persons: List[dict],          # [{"track_id": int, "bbox": [x1,y1,x2,y2]}]
    ) -> List[dict]:
        events: List[dict] = []

        for p in tracked_persons:
            self.person_last_seen[p["track_id"]] = (list(p["bbox"]), frame_idx)

        for aid, asset in registered_assets.items():
            if aid not in presence:
                continue  # 매칭 불가 자산은 판정 제외
            rec = self.records.setdefault(aid, _AssetRecord())
            if rec.state == TheftState.THEFT_ALERT:
                continue

            ax, ay = box_center(asset["bbox"])
            is_present = presence[aid]

            # 가장 가까운 사람 / 가려짐 여부
            nearest_id, nearest_d = None, float("inf")
            for p in tracked_persons:
                d = point_to_box_distance(ax, ay, p["bbox"])
                if d < nearest_d:
                    nearest_id, nearest_d = p["track_id"], d
            is_near = nearest_id is not None and nearest_d <= self.interact_dist
            is_occluded = nearest_id is not None and nearest_d <= self.occlusion_margin

            # (1) 접촉 기록: 물품이 보이거나, 사람 몸/손이 덮고 있을 때만 갱신
            #     (물품이 사라진 뒤 지나가는 행인이 용의자로 덮어쓰지 않도록)
            if is_near and (is_present or is_occluded):
                person_bbox = next(p["bbox"] for p in tracked_persons if p["track_id"] == nearest_id)
                rec.interactor_id = nearest_id
                rec.interactor_frame = frame_idx
                rec.interactor_door_dist = self._door_dist(person_bbox)
                if rec.state == TheftState.SAFE:
                    rec.state = TheftState.INTERACTING

            # (2) 결손 카운트
            if is_present:
                rec.missing_frames = 0
                rec.missing_since_frame = None
                if rec.state in (TheftState.MISSING_SUSPECT, TheftState.INTERACTING):
                    rec.state = TheftState.SAFE  # 되돌려놓음/단순 조작
                continue

            if is_occluded:
                continue  # 가려진 구간은 결손 증거로 쓰지 않음

            has_suspect = (rec.interactor_id is not None
                           and frame_idx - rec.interactor_frame <= self.memory_frames)
            if not has_suspect:
                rec.missing_frames = 0
                continue  # 접촉 이력 없는 소멸은 도난으로 보지 않음 (조명/노이즈)

            rec.missing_frames += 1
            if rec.missing_frames < self.confirm_missing_frames:
                continue

            if rec.state != TheftState.MISSING_SUSPECT:
                rec.state = TheftState.MISSING_SUSPECT
                rec.missing_since_frame = frame_idx

            # (3) 접촉자의 이탈/출입문 방향 이동 판정
            suspect_bbox, seen_frame = self.person_last_seen.get(rec.interactor_id, (None, -10 ** 9))
            if suspect_bbox is None or frame_idx - seen_frame > self.lost_track_frames:
                continue
            sx, sy = box_center(suspect_bbox)
            departure = math.hypot(sx - ax, sy - ay)
            door_d = self._door_dist(suspect_bbox)
            heading_door = (
                door_d <= self.door_trigger_dist
                or (rec.interactor_door_dist is not None
                    and door_d <= rec.interactor_door_dist - self.approach_px)
            )
            if heading_door and departure >= self.min_departure_px:
                rec.state = TheftState.THEFT_ALERT
                events.append({
                    "asset_id": aid,
                    "class_name": asset["class"],
                    "bbox": list(asset["bbox"]),
                    "suspect_id": rec.interactor_id,
                    "suspect_bbox": suspect_bbox,
                    "frame_idx": frame_idx,
                    "door_distance": round(door_d, 1),
                    "departure_px": round(departure, 1),
                })
        return events
