"""
[개인 소지품 절도 판정 엔진 - core/personal_item_tracker.py]

공용 자산(DynamicTheftDetector)과 달리 개인 소지품은 '주인'이 있다.
  - 주인이 직접 가져가는 것  -> 정상 (경보 없음)
  - 주인이 자리를 비운 사이 '제3자'가 접촉 -> 물품 소멸 -> 출입문 방향 이동  => PERSONAL_THEFT
  - 주인이 떠난 채 방치      -> PERSONAL_ABANDONED (분실물)

핵심 방어 로직 (엣지 케이스)
  1) 소유권 자동 부여     : 물품 곁에 일정 시간 머문 ByteTrack ID를 주인으로 등록
  2) 주인 ID 스위칭 보정  : 주인 트랙이 끊긴 직후 물품 근처에서 새로 생긴 ID는 같은 주인으로 승계
  3) 주인이 근처에 있으면 판정 보류 (주인이 집었는지 제3자가 집었는지 모호한 상황은 경보하지 않음)
  4) 행인 필터            : 제3자는 touch_sec 이상 연속 접촉해야 용의자로 기록
  5) 되돌려 놓음 복구     : 물품이 다시 보이면 의심 상태 해제
  6) 소멸 후 행인 오탐 차단: 용의자는 물품이 보이거나 가려져 있을 때만 갱신
  7) 가려짐 구간은 소멸 증거에서 제외
  8) 출입문 방향 이동이 아니면 경보 대신 PERSONAL_MISSING(수동 확인용)만 발행

시간 단위는 모두 프레임 수 / fps 로 환산한다.
"""
import math
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from core.interaction_tracker import box_center, point_to_box_distance


class PersonalState(Enum):
    UNOWNED = "UNOWNED"                    # 주인 미확정
    OWNED = "OWNED"                        # 주인이 곁에 있음
    OWNER_AWAY = "OWNER_AWAY"              # 주인이 자리를 비움
    ABANDONED = "ABANDONED"                # 오래 방치 -> 분실물 확정
    THIRD_PARTY_TOUCH = "THIRD_PARTY_TOUCH"  # 제3자가 접촉 중
    MISSING_SUSPECT = "MISSING_SUSPECT"    # 물품 소멸, 제3자 이동 경로 감시
    OWNER_RETRIEVED = "OWNER_RETRIEVED"    # 주인이 직접 회수
    THEFT_ALERT = "THEFT_ALERT"            # 개인 소지품 절도 확정


class _ItemRecord:
    def __init__(self):
        self.state = PersonalState.UNOWNED
        self.owner_id: Optional[int] = None
        self.cand_id: Optional[int] = None
        self.cand_frames = 0
        self.owner_last_near = -10 ** 9
        self.owner_last_seen = -10 ** 9
        self.toucher_id: Optional[int] = None
        self.streak_id: Optional[int] = None
        self.streak_frames = 0
        self.toucher_frame = -10 ** 9
        self.toucher_door_dist: Optional[float] = None
        self.missing_frames = 0
        self.missing_since: Optional[int] = None
        self.abandon_emitted = False
        self.unresolved_emitted = False


class PersonalItemTheftDetector:
    def __init__(
        self,
        exit_door_bbox: Tuple[int, int, int, int],
        fps: float = 24.0,
        own_dist: float = 100.0,
        own_sec: float = 1.5,
        interact_dist: float = 150.0,
        away_sec: float = 1.5,
        touch_sec: float = 1.0,
        confirm_missing_sec: float = 0.6,
        interaction_memory_sec: float = 10.0,
        abandon_sec: float = 10.0,
        unresolved_sec: float = 30.0,
        reassoc_sec: float = 2.0,
        reassoc_dist: float = 150.0,
        occlusion_margin: float = 0.0,
        door_trigger_dist: float = 250.0,
        approach_px: float = 150.0,
        min_departure_px: float = 150.0,
        lost_track_sec: float = 2.0,
    ):
        self.exit_door = exit_door_bbox
        f = lambda sec: max(1, int(sec * fps))
        self.own_dist = own_dist
        self.own_frames = f(own_sec)
        self.interact_dist = interact_dist
        self.away_frames = f(away_sec)
        self.touch_frames = f(touch_sec)
        self.confirm_frames = f(confirm_missing_sec)
        self.memory_frames = f(interaction_memory_sec)
        self.abandon_frames = f(abandon_sec)
        self.unresolved_frames = f(unresolved_sec)
        self.reassoc_frames = f(reassoc_sec)
        self.reassoc_dist = reassoc_dist
        self.occlusion_margin = occlusion_margin
        self.door_trigger_dist = door_trigger_dist
        self.approach_px = approach_px
        self.min_departure_px = min_departure_px
        self.lost_track_frames = f(lost_track_sec)

        self.records: Dict[str, _ItemRecord] = {}
        self.known_ids: Set[int] = set()
        self.person_last_seen: Dict[int, Tuple[List[float], int]] = {}

    # ------------------------------------------------------------------
    def _door_dist(self, bbox) -> float:
        cx, cy = box_center(bbox)
        return point_to_box_distance(cx, cy, self.exit_door)

    def get_state(self, item_id: str) -> PersonalState:
        rec = self.records.get(item_id)
        return rec.state if rec else PersonalState.UNOWNED

    def get_owner(self, item_id: str) -> Optional[int]:
        rec = self.records.get(item_id)
        return rec.owner_id if rec else None

    # ------------------------------------------------------------------
    def update(
        self,
        frame_idx: int,
        items: Dict[str, dict],            # {id: {"class", "bbox"}}
        presence: Dict[str, bool],         # {id: 물품이 현재 위치에 있는가}
        tracked_persons: List[dict],       # [{"track_id", "bbox"}]
    ) -> List[dict]:
        events: List[dict] = []
        ids_now = {p["track_id"] for p in tracked_persons}
        new_ids = ids_now - self.known_ids
        self.known_ids |= ids_now
        for p in tracked_persons:
            self.person_last_seen[p["track_id"]] = (list(p["bbox"]), frame_idx)
        bbox_of = {p["track_id"]: p["bbox"] for p in tracked_persons}

        for aid, item in items.items():
            if aid not in presence:
                continue
            rec = self.records.setdefault(aid, _ItemRecord())
            if rec.state == PersonalState.THEFT_ALERT:
                continue

            ax, ay = box_center(item["bbox"])
            dists = {pid: point_to_box_distance(ax, ay, b) for pid, b in bbox_of.items()}
            present = presence[aid]

            # --- (2) 주인 ID 스위칭 보정 ---------------------------------
            if rec.owner_id is not None:
                if rec.owner_id in ids_now:
                    rec.owner_last_seen = frame_idx
                elif frame_idx - rec.owner_last_seen <= self.reassoc_frames:
                    cands = [i for i in new_ids if dists.get(i, 1e9) <= self.reassoc_dist]
                    if cands:
                        rec.owner_id = min(cands, key=lambda i: dists[i])
                        rec.owner_last_seen = frame_idx
                        rec.owner_last_near = frame_idx

            # --- (1) 소유권 자동 부여 -----------------------------------
            if rec.owner_id is None:
                if present and dists:
                    nid = min(dists, key=dists.get)
                    if dists[nid] <= self.own_dist:
                        if rec.cand_id == nid:
                            rec.cand_frames += 1
                        else:
                            rec.cand_id, rec.cand_frames = nid, 1
                        if rec.cand_frames >= self.own_frames:
                            rec.owner_id = nid
                            rec.owner_last_near = rec.owner_last_seen = frame_idx
                            rec.state = PersonalState.OWNED
                    else:
                        rec.cand_id, rec.cand_frames = None, 0
                else:
                    rec.cand_id, rec.cand_frames = None, 0
                continue

            owner_d = dists.get(rec.owner_id)
            owner_near = owner_d is not None and owner_d <= self.interact_dist
            if owner_near:
                rec.owner_last_near = frame_idx
            owner_away = frame_idx - rec.owner_last_near >= self.away_frames

            others = {pid: d for pid, d in dists.items() if pid != rec.owner_id}
            other_id = min(others, key=others.get) if others else None
            other_near = other_id is not None and others[other_id] <= self.interact_dist
            occluded = bool(dists) and min(dists.values()) <= self.occlusion_margin

            # ================= 물품이 보이는 경우 =================
            if present:
                rec.missing_frames = 0
                rec.missing_since = None
                if owner_near:
                    # (3)(5) 주인이 곁에 있으면 모든 의심 해제
                    rec.state = PersonalState.OWNED
                    rec.toucher_id = None
                    rec.streak_id, rec.streak_frames = None, 0
                elif owner_away:
                    if other_near:
                        # (4) 제3자 연속 접촉 시간 누적 (행인 필터)
                        if rec.streak_id == other_id:
                            rec.streak_frames += 1
                        else:
                            rec.streak_id, rec.streak_frames = other_id, 1
                        if rec.streak_frames >= self.touch_frames:
                            rec.toucher_id = other_id
                            rec.toucher_frame = frame_idx
                            rec.toucher_door_dist = self._door_dist(bbox_of[other_id])
                            rec.state = PersonalState.THIRD_PARTY_TOUCH
                    else:
                        rec.streak_id, rec.streak_frames = None, 0
                        if rec.state in (PersonalState.THIRD_PARTY_TOUCH, PersonalState.MISSING_SUSPECT):
                            rec.state = PersonalState.OWNER_AWAY
                            rec.toucher_id = None
                    if rec.state in (PersonalState.OWNED, PersonalState.UNOWNED):
                        rec.state = PersonalState.OWNER_AWAY
                    if (rec.state == PersonalState.OWNER_AWAY and not rec.abandon_emitted
                            and frame_idx - rec.owner_last_near >= self.abandon_frames):
                        rec.abandon_emitted = True
                        rec.state = PersonalState.ABANDONED
                        events.append(self._event("PERSONAL_ABANDONED", aid, item, rec, frame_idx))
                continue

            # ================= 물품이 사라진 경우 =================
            if owner_near:
                # 주인이 직접 집었거나 손으로 가린 것 -> 도난 아님
                rec.state = PersonalState.OWNER_RETRIEVED
                rec.toucher_id = None
                rec.missing_frames = 0
                continue
            if occluded:
                continue  # (7) 가려진 구간은 소멸 증거로 쓰지 않음
            has_suspect = (rec.toucher_id is not None
                           and (rec.state == PersonalState.MISSING_SUSPECT
                                or frame_idx - rec.toucher_frame <= self.memory_frames))
            if not has_suspect or not owner_away:
                rec.missing_frames = 0
                continue  # 제3자 접촉 이력이 없는 소멸은 절도로 보지 않음

            rec.missing_frames += 1
            if rec.missing_frames < self.confirm_frames:
                continue
            if rec.state != PersonalState.MISSING_SUSPECT:
                rec.state = PersonalState.MISSING_SUSPECT
                rec.missing_since = frame_idx

            suspect_bbox, seen = self.person_last_seen.get(rec.toucher_id, (None, -10 ** 9))
            if suspect_bbox is not None:
                sx, sy = box_center(suspect_bbox)
                departure = math.hypot(sx - ax, sy - ay)
                door_d = self._door_dist(suspect_bbox)
                recently_seen = frame_idx - seen <= self.lost_track_frames
                exited_via_door = door_d <= self.door_trigger_dist  # 출입문에서 사라진 경우도 허용
                heading = (door_d <= self.door_trigger_dist
                           or (rec.toucher_door_dist is not None
                               and door_d <= rec.toucher_door_dist - self.approach_px))
                if heading and departure >= self.min_departure_px and (recently_seen or exited_via_door):
                    rec.state = PersonalState.THEFT_ALERT
                    ev = self._event("PERSONAL_THEFT", aid, item, rec, frame_idx)
                    ev.update({
                        "suspect_id": rec.toucher_id,
                        "suspect_bbox": suspect_bbox,
                        "door_distance": round(door_d, 1),
                        "departure_px": round(departure, 1),
                    })
                    events.append(ev)
                    continue

            # (8) 출입문 방향 이탈이 확인되지 않은 채 오래 지속 -> 수동 확인 요청
            if (rec.missing_since is not None and not rec.unresolved_emitted
                    and frame_idx - rec.missing_since >= self.unresolved_frames):
                rec.unresolved_emitted = True
                events.append(self._event("PERSONAL_MISSING", aid, item, rec, frame_idx))

        return events

    @staticmethod
    def _event(event_type: str, aid: str, item: dict, rec: _ItemRecord, frame_idx: int) -> dict:
        return {
            "event_type": event_type,
            "asset_id": aid,
            "class_name": item["class"],
            "bbox": list(item["bbox"]),
            "owner_id": rec.owner_id,
            "suspect_id": rec.toucher_id,
            "frame_idx": frame_idx,
        }
