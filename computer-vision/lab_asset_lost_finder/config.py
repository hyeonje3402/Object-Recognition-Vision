"""
[설정 모듈 - config.py]
실습실 구역(ROI) 좌표, 클래스 정의, 상태 머신 타이머 임계값 설정
"""
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class ROIConfig:
    # (x1, y1, x2, y2) 정규화 좌표 (0.0 ~ 1.0) 또는 픽셀 좌표
    # 카메라 해상도가 달라져도 유연하게 적용되도록 0.0 ~ 1.0 비율 좌표 사용
    PODIUM_DESK: Tuple[float, float, float, float] = (0.05, 0.40, 0.40, 0.85)   # ROI 1: 공용 교탁/PC 영역
    DOOR_EXIT: Tuple[float, float, float, float]   = (0.75, 0.10, 0.98, 0.90)   # ROI 2: 실습실 출입문 영역
    STUDENT_DESK: Tuple[float, float, float, float]= (0.35, 0.45, 0.85, 0.95)   # ROI 3: 학생 좌석/책상 영역

# 클래스 정의
TARGET_CLASSES = {
    "PUBLIC_ASSETS": ["mouse", "keyboard", "cable", "hdmi", "remote"], # 공용 기자재
    "PERSONAL_ITEMS": ["cell phone", "laptop", "tablet", "ipad"],       # 중대형 개인 물품
    "SMALL_PERSONAL_ITEMS": ["small_personal_item", "ring", "tint", "keychain", "earbuds"], # 초소형 분실물
    "PERSON": ["person"]                                               # 사람(학생/교수/외부인)
}

# COCO 데이터셋 기준 클래스 매핑 (실습 및 테스트용)
# 실제 커스텀 학습 시 'hdmi', 'small_personal_item' 등이 클래스로 추가됨
COCO_CLASS_MAP = {
    0: "person",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    63: "laptop",
    # 커스텀 모델 클래스 ID 예시 (향후 학습 시 매핑)
    80: "small_personal_item"
}

# 타이머 설정 (초 단위)
# 실전 환경: 공용 기자재 부재 허용 시간 180초(3분), 분실물 방치 판정 300초(5분)
# 테스트/데모 환경: 빠른 시연을 위해 짧게 조절 가능
TIMER_CONFIG = {
    "ASSET_MISSING_TIMEOUT_SEC": 10.0,       # 공용 기자재 실종 허용 시간 (데모용 10초)
    "LOST_ITEM_TIMEOUT_SEC": 8.0,            # 사람 없이 방치된 시간 (데모용 8초)
    "PERSON_PROXIMITY_PIXELS": 150           # 소지자와 물품 간의 유효 거리(픽셀)
}
