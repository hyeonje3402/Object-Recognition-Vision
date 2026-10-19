"""
[상태 머신 모듈 - core/state_machine.py]
1. 공용 기자재 반출 방지 상태 머신 (AssetSecurityStateMachine)
2. 개인 분실물 방치 탐지 상태 머신 (LostPropertyStateMachine)
"""
import time
from enum import Enum
from typing import Optional, Dict, Any

class AssetState(Enum):
    SAFE = "SAFE"                          # 정상 (교탁 ROI에 상시 존재)
    IN_USE_PODIUM = "IN_USE_PODIUM"        # [4주차 문제 2 해결] 교탁에서 정상 사용/조작 중 (오경보 방지)
    WARNING = "WARNING"                    # 주의 (기자재 사라짐 / 이동 중 타이머 작동)
    THEFT_ALERT = "THEFT_ALERT"            # 반출 경고 (타이머 초과 OR 출입문 ROI 사람 감지)

class LostState(Enum):
    ATTENDED = "ATTENDED"          # 정상 (물품 주변에 사람이 존재함)
    UNATTENDED = "UNATTENDED"      # 주의 (사람이 이탈하여 방치 타이머 시작)
    LOST_CONFIRMED = "LOST"        # 분실물 확정 (설정 시간 이상 사람 부재)

class AssetSecurityStateMachine:
    """
    [룰 1 고도화 - 4주차 문제 2 해결]
    - 교탁 마우스가 손에 가려져 일시적으로 안 보이더라도 교탁에 교수/발표자가 있으면
      'IN_USE_PODIUM'으로 판정하여 오경보를 0%로 억제
    - 교탁에 사람이 없는 상태에서 마우스가 사라지고 출입문으로 향할 때만 'THEFT_ALERT'
    """
    def __init__(self, asset_name: str, timeout_sec: float = 10.0):
        self.asset_name = asset_name
        self.state = AssetState.SAFE
        self.timeout_sec = timeout_sec
        self.missing_start_time: Optional[float] = None
        self.elapsed_missing_time: float = 0.0

    def update(self, is_detected_in_podium: bool, is_person_at_podium: bool, is_person_at_door: bool) -> AssetState:
        current_time = time.time()

        # 1. 교탁 위에 마우스가 그대로 감지되는 경우
        if is_detected_in_podium:
            self.state = AssetState.SAFE
            self.missing_start_time = None
            self.elapsed_missing_time = 0.0
            return self.state

        # 2. 마우스는 안 보이지만 교탁에 발표자/교수가 서서 조작 중인 경우 (가려짐 현상)
        if is_person_at_podium and not is_person_at_door:
            self.state = AssetState.IN_USE_PODIUM
            # 정상 수업 중이므로 타이머 일시 보류/리셋
            self.missing_start_time = current_time
            self.elapsed_missing_time = 0.0
            return self.state

        # 3. 교탁에 사람도 없는데 마우스가 사라진 경우 (실제 반출 의심)
        if self.missing_start_time is None:
            self.missing_start_time = current_time

        self.elapsed_missing_time = current_time - self.missing_start_time

        # 출입문(Door)에 사람이 감지되면 즉시 긴급 반출 경보
        if is_person_at_door:
            self.state = AssetState.THEFT_ALERT
            return self.state

        # 타이머 만료 시 반출 경보
        if self.elapsed_missing_time >= self.timeout_sec:
            self.state = AssetState.THEFT_ALERT
        else:
            self.state = AssetState.WARNING

        return self.state

    def get_status_info(self) -> Dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "state": self.state.value,
            "elapsed_missing_sec": round(self.elapsed_missing_time, 1),
            "remaining_sec": max(0.0, round(self.timeout_sec - self.elapsed_missing_time, 1))
        }


class LostPropertyStateMachine:
    """
    [룰 2] 개인 물품 (아이패드, 에어팟, 틴트/반지 등 small_personal_item) 상태 머신
    - 책상 영역(ROI 3)에 물건이 감지됨
    - 물건 주변(반경 N 픽셀)에 사람(Person)의 Bounding Box가 존재하는지 검사
    - 사람이 없으면 타이머 가동 -> 일정 시간 초과 시 '분실물 방치(Lost)' 전환
    """
    def __init__(self, item_id: int, item_class: str, timeout_sec: float = 8.0):
        self.item_id = item_id
        self.item_class = item_class
        self.state = LostState.ATTENDED
        self.timeout_sec = timeout_sec
        self.unattended_start_time: Optional[float] = None
        self.elapsed_unattended_time: float = 0.0

    def update(self, is_person_nearby: bool) -> LostState:
        current_time = time.time()

        if is_person_nearby:
            # 사람이 근처에 있으면 정상 소지 상태로 복귀
            self.state = LostState.ATTENDED
            self.unattended_start_time = None
            self.elapsed_unattended_time = 0.0
            return self.state

        # 사람이 주변에 없는 경우 (방치 상태)
        if self.unattended_start_time is None:
            self.unattended_start_time = current_time

        self.elapsed_unattended_time = current_time - self.unattended_start_time

        # 타이머 만료 시 분실물 상태 확정
        if self.elapsed_unattended_time >= self.timeout_sec:
            self.state = LostState.LOST_CONFIRMED
        else:
            self.state = LostState.UNATTENDED

        return self.state

    def get_status_info(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_class": self.item_class,
            "state": self.state.value,
            "unattended_sec": round(self.elapsed_unattended_time, 1),
            "remaining_sec": max(0.0, round(self.timeout_sec - self.elapsed_unattended_time, 1))
        }
