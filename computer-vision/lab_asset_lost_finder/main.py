"""
[실습실 기자재 반출 방지 및 분실물 탐지 시스템 - main.py]
실시간 프레임 추론, 2가지 상태 머신 룰 평가, HUD 시각화 및 이벤트 발행
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

# 현재 디렉토리 모듈 import
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from config import ROIConfig, TIMER_CONFIG, COCO_CLASS_MAP
from core.state_machine import AssetSecurityStateMachine, LostPropertyStateMachine, AssetState, LostState
from core.vision_engine import VisionEngine

class LabSecuritySystem:
    def __init__(self, model_path: str = "yolo11n.pt", source: Any = 0):
        """
        :param model_path: YOLO 모델 경로
        :param source: 0(웹캠) 또는 비디오 파일 경로
        """
        # 상위 디렉토리 모델 경로 탐색
        parent_model = CURRENT_DIR.parent / "yolo11n.pt"
        if parent_model.exists():
            model_path = str(parent_model)
        elif (CURRENT_DIR.parent / "yolov8n.pt").exists():
            model_path = str(CURRENT_DIR.parent / "yolov8n.pt")

        self.vision = VisionEngine(model_path=model_path, use_clahe=True)
        self.source = source
        self.roi_cfg = ROIConfig()

        # [상태 머신 인스턴스 초기화]
        # 1. 공용 기자재: 교탁 마우스 & HDMI 케이블
        self.mouse_asset_sm = AssetSecurityStateMachine(
            asset_name="Public Mouse",
            timeout_sec=TIMER_CONFIG["ASSET_MISSING_TIMEOUT_SEC"]
        )

        # 2. 책상 위 개인 분실물 추적 딕셔너리: {item_id: LostPropertyStateMachine}
        self.lost_items_sm: Dict[int, LostPropertyStateMachine] = {}

    def get_pixel_rois(self, width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
        """정규화된 ROI 비율을 프레임 해상도 기준 픽셀 좌표로 변환"""
        return {
            "PODIUM": (
                int(self.roi_cfg.PODIUM_DESK[0] * width),
                int(self.roi_cfg.PODIUM_DESK[1] * height),
                int(self.roi_cfg.PODIUM_DESK[2] * width),
                int(self.roi_cfg.PODIUM_DESK[3] * height),
            ),
            "DOOR": (
                int(self.roi_cfg.DOOR_EXIT[0] * width),
                int(self.roi_cfg.DOOR_EXIT[1] * height),
                int(self.roi_cfg.DOOR_EXIT[2] * width),
                int(self.roi_cfg.DOOR_EXIT[3] * height),
            ),
            "STUDENT_DESK": (
                int(self.roi_cfg.STUDENT_DESK[0] * width),
                int(self.roi_cfg.STUDENT_DESK[1] * height),
                int(self.roi_cfg.STUDENT_DESK[2] * width),
                int(self.roi_cfg.STUDENT_DESK[3] * height),
            ),
        }

    def emit_event(self, event_type: str, payload: Dict[str, Any]):
        """
        [프론트엔드 연동용 이벤트 디스패처]
        - 향후 WebSocket(FastAPI) 또는 REST API로 알림을 브로드캐스팅하는 인터페이스
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [EVENT::{event_type}] {payload}"
        print(log_msg)

    def draw_hud(self, frame: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]):
        """CCTV 스타일 HUD 대시보드 및 경보 배너 렌더링"""
        h, w, _ = frame.shape

        # 1. ROI 영역 박스 그리기
        # ROI 1: 교탁 (하늘색)
        p_roi = rois["PODIUM"]
        cv2.rectangle(frame, (p_roi[0], p_roi[1]), (p_roi[2], p_roi[3]), (255, 200, 0), 2)
        cv2.putText(frame, "ROI 1: PODIUM (PUBLIC ASSET)", (p_roi[0] + 5, p_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # ROI 2: 출입문 (자주색)
        d_roi = rois["DOOR"]
        cv2.rectangle(frame, (d_roi[0], d_roi[1]), (d_roi[2], d_roi[3]), (200, 0, 200), 2)
        cv2.putText(frame, "ROI 2: EXIT DOOR", (d_roi[0] + 5, d_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 200), 1)

        # ROI 3: 학생 책상 (연청색)
        s_roi = rois["STUDENT_DESK"]
        cv2.rectangle(frame, (s_roi[0], s_roi[1]), (s_roi[2], s_roi[3]), (0, 255, 255), 2)
        cv2.putText(frame, "ROI 3: DESK (PERSONAL/LOST ITEM)", (s_roi[0] + 5, s_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 2. 상단 상태 요약 헤더 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # 공용 기자재 상태 표시
        mouse_info = self.mouse_asset_sm.get_status_info()
        asset_state = mouse_info["state"]
        if asset_state == AssetState.SAFE.value:
            color = (0, 255, 0)
            text = "[ASSET: SAFE] Public Mouse Present on Podium"
        elif asset_state == AssetState.WARNING.value:
            color = (0, 200, 255) # 노란색
            text = f"[ASSET: WARNING] Mouse Missing! Timer: {mouse_info['remaining_sec']}s left"
        else:
            color = (0, 0, 255) # 빨간색
            text = "[ASSET: THEFT ALERT] Public Equipment Removed or Unreturned!"
        cv2.putText(frame, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 분실물 상태 표시
        lost_count = sum(1 for sm in self.lost_items_sm.values() if sm.state == LostState.LOST_CONFIRMED)
        unattended_count = sum(1 for sm in self.lost_items_sm.values() if sm.state == LostState.UNATTENDED)
        lost_text = f"LOST ITEMS: {lost_count} Confirmed | {unattended_count} In Grace Period"
        cv2.putText(frame, lost_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """단일 프레임 처리 파이프라인"""
        h, w, _ = frame.shape
        rois = self.get_pixel_rois(w, h)

        # 1. YOLO 객체 탐지 (기본 프레임 추론)
        detections = self.vision.detect(frame, conf=0.25)

        # 2. 책상 영역(ROI 3)에 대해 초소형 객체(Small Object) 고해상도 Crop 2차 추론 적용
        small_detections = self.vision.detect_roi_cropped(frame, rois["STUDENT_DESK"], conf=0.15)
        # 중복 방지하며 결합
        detections.extend(small_detections)

        # 탐지 객체 분류
        persons = []
        podium_assets = []
        desk_personal_items = []

        is_person_at_door = False

        for det in detections:
            cls_name = det["class_name"].lower()
            box = det["box"]

            # 사람 판별
            if cls_name == "person":
                persons.append(det)
                # 출입문(ROI 2) 영역에 사람이 있는지 확인
                if self.vision.is_inside_roi(box, rois["DOOR"], threshold=0.3):
                    is_person_at_door = True
                continue

            # 공용 기자재 판별 (마우스, 키보드 등)
            if cls_name in ["mouse", "keyboard", "cable"]:
                if self.vision.is_inside_roi(box, rois["PODIUM"], threshold=0.2):
                    podium_assets.append(det)

            # 개인 물품 판별 (스마트폰, 랩탑, 에어팟, 틴트, small_personal_item 등)
            if cls_name in ["cell phone", "laptop", "small_personal_item", "bottle", "cup"]:
                if self.vision.is_inside_roi(box, rois["STUDENT_DESK"], threshold=0.2):
                    desk_personal_items.append(det)

        # ==========================================
        # [룰 1] 공용 기자재 상태 머신 업데이트
        # ==========================================
        is_mouse_detected = len(podium_assets) > 0
        prev_asset_state = self.mouse_asset_sm.state
        new_asset_state = self.mouse_asset_sm.update(
            is_detected_in_podium=is_mouse_detected,
            is_person_at_door=is_person_at_door
        )

        # 상태 변화 시 이벤트 발행
        if prev_asset_state != new_asset_state:
            self.emit_event("ASSET_STATE_CHANGED", {
                "asset": "Public Mouse",
                "from": prev_asset_state.value,
                "to": new_asset_state.value,
                "person_at_door": is_person_at_door
            })

        # ==========================================
        # [룰 2] 개인 분실물 상태 머신 업데이트
        # ==========================================
        current_item_ids = set()
        proximity_threshold = TIMER_CONFIG["PERSON_PROXIMITY_PIXELS"]

        for idx, item in enumerate(desk_personal_items):
            item_id = idx + 1
            current_item_ids.add(item_id)
            item_center = item["center"]

            # 물품 주변에 사람 존재 여부 검사 (유클리디안 거리 기준)
            has_person_nearby = False
            for p in persons:
                p_box = p["box"]
                # 사람의 바운딩 박스 중심점 또는 하단과의 거리
                p_center = ((p_box[0] + p_box[2]) // 2, (p_box[1] + p_box[3]) // 2)
                dist = self.vision.calculate_distance(item_center, p_center)
                if dist <= proximity_threshold:
                    has_person_nearby = True
                    break

            if item_id not in self.lost_items_sm:
                self.lost_items_sm[item_id] = LostPropertyStateMachine(
                    item_id=item_id,
                    item_class=item["class_name"],
                    timeout_sec=TIMER_CONFIG["LOST_ITEM_TIMEOUT_SEC"]
                )

            sm = self.lost_items_sm[item_id]
            prev_lost_state = sm.state
            new_lost_state = sm.update(is_person_nearby=has_person_nearby)

            if prev_lost_state != new_lost_state:
                self.emit_event("LOST_STATE_CHANGED", {
                    "item_id": item_id,
                    "class": item["class_name"],
                    "from": prev_lost_state.value,
                    "to": new_lost_state.value
                })

            # 시각화 박스 렌더링
            bx1, by1, bx2, by2 = item["box"]
            if new_lost_state == LostState.ATTENDED:
                box_col = (0, 255, 0) # 초록
                tag = f"Personal: {item['class_name']}"
            elif new_lost_state == LostState.UNATTENDED:
                box_col = (0, 200, 255) # 노랑
                tag = f"Grace: {sm.get_status_info()['remaining_sec']}s"
            else:
                box_col = (0, 0, 255) # 빨강
                tag = f"[LOST CONFIRMED] {item['class_name']}"

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_col, 2)
            cv2.putText(frame, tag, (bx1, max(20, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 2)

        # 사라진 아이템 상태 머신 정리
        for dead_id in list(self.lost_items_sm.keys()):
            if dead_id not in current_item_ids:
                del self.lost_items_sm[dead_id]

        # 사람 박스 렌더링
        for p in persons:
            px1, py1, px2, py2 = p["box"]
            p_col = (255, 0, 0) if not is_person_at_door else (0, 0, 255)
            cv2.rectangle(frame, (px1, py1), (px2, py2), p_col, 2)
            cv2.putText(frame, f"Person ({p['conf']:.2f})", (px1, max(20, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_col, 1)

        # HUD 렌더링
        self.draw_hud(frame, rois)
        return frame

    def run(self):
        """실행 루프"""
        print(f"[LabSecuritySystem] 시스템 시작... 입력 소스: {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[오류] 비디오 소스를 열 수 없습니다: {self.source}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow("Lab Asset & Lost Property Security System", processed_frame)

            # ESC 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 0(기본 웹캠) 또는 비디오 파일 경로 지정
    system = LabSecuritySystem(source=0)
    system.run()
