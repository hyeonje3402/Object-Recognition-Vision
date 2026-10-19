"""
[비전 처리 엔진 - core/vision_engine.py]
YOLO 기반 추론 및 초소형 객체(Small Object) 검출율 향상 파이프라인
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

class VisionEngine:
    def __init__(self, model_path: str = "yolov8n.pt", use_clahe: bool = True):
        """
        초기화 및 모델 로드
        :param model_path: YOLO 모델 가중치 경로
        :param use_clahe: 적응형 히스토그램 평활화(CLAHE) 전처리 적용 여부
        """
        print(f"[VisionEngine] 모델 로드 중: {model_path}")
        self.model = YOLO(model_path)
        self.use_clahe = use_clahe
        # 틴트, 반지 등 초소형 물품의 명암 대비 극대화를 위한 CLAHE 필터
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        # [4주차 문제 1 해결: FPS 최적화를 위한 모션 감지기]
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25, detectShadows=False)
        self.last_crop_detections = []
        self.frame_counter = 0

    def preprocess_for_small_objects(self, image: np.ndarray) -> np.ndarray:
        """CLAHE 적응형 명암 대비 증폭 전처리"""
        if not self.use_clahe:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def detect(self, frame: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> List[Dict[str, Any]]:
        """기본 프레임 전체 1차 추론"""
        processed_frame = self.preprocess_for_small_objects(frame)
        results = self.model(processed_frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            names = self.model.names
            for box, score, cls_id in zip(boxes, confs, clss):
                detections.append({
                    "box": box,
                    "conf": float(score),
                    "class_id": int(cls_id),
                    "class_name": names.get(cls_id, str(cls_id)),
                    "center": ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                })
        return detections

    def check_roi_motion(self, roi_img: np.ndarray, motion_thresh: float = 0.015) -> bool:
        """
        ROI 내 움직임(Motion) 비율 검사
        - 책상에 손이 오가거나 물건이 움직일 때만 True 반환
        - 정적인 상태에서는 불필요한 2차 YOLO 추론을 건너뛰어 FPS를 3배 이상 향상
        """
        if roi_img.size == 0:
            return False
        fg_mask = self.motion_detector.apply(roi_img)
        motion_ratio = np.mean(fg_mask > 0)
        return motion_ratio > motion_thresh

    def detect_roi_cropped_optimized(self, frame: np.ndarray, roi_box: Tuple[int, int, int, int], conf: float = 0.15) -> Tuple[List[Dict[str, Any]], bool]:
        """
        [4주차 문제 1 해결 핵심: Motion-Triggered Sliced Inference]
        - 프레임 카운트 및 모션 감지를 결합하여 2차 추론 빈도를 지능적으로 제어
        - 모션이 없으면 이전 검출 결과를 캐싱(Reuse)하여 30+ FPS 달성
        """
        self.frame_counter += 1
        rx1, ry1, rx2, ry2 = roi_box
        roi_img = frame[ry1:ry2, rx1:rx2]
        if roi_img.size == 0:
            return [], False

        has_motion = self.check_roi_motion(roi_img)
        # 모션이 감지되었거나, 최소 15프레임(약 0.5초) 주기로 1회 강제 동기화
        should_infer = has_motion or (self.frame_counter % 15 == 0)

        if not should_infer and self.last_crop_detections:
            # 캐싱된 결과 반환 (연산 비용 0ms)
            return self.last_crop_detections, False

        # 모션 발생 시 고해상도 Crop 2차 추론 실행
        enhanced_roi = self.preprocess_for_small_objects(roi_img)
        results = self.model(enhanced_roi, conf=conf, imgsz=640, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            names = self.model.names

            for box, score, cls_id in zip(boxes, confs, clss):
                gx1 = rx1 + box[0]
                gy1 = ry1 + box[1]
                gx2 = rx1 + box[2]
                gy2 = ry1 + box[3]
                class_name = names.get(cls_id, str(cls_id))

                detections.append({
                    "box": np.array([gx1, gy1, gx2, gy2]),
                    "conf": float(score),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "center": ((gx1 + gx2) // 2, (gy1 + gy2) // 2)
                })

        self.last_crop_detections = detections
        return detections, True

    @staticmethod
    def is_inside_roi(box: np.ndarray, roi: Tuple[int, int, int, int], threshold: float = 0.4) -> bool:
        """
        객체의 바운딩 박스가 지정된 ROI 영역 내에 포함되어 있는지 판별
        """
        bx1, by1, bx2, by2 = box
        rx1, ry1, rx2, ry2 = roi

        # 교차 영역 계산
        ix1 = max(bx1, rx1)
        iy1 = max(by1, ry1)
        ix2 = min(bx2, rx2)
        iy2 = min(by2, ry2)

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        box_area = (bx2 - bx1) * (by2 - by1)

        # 박스 면적의 일정 비율 이상이 ROI 안에 들어오면 포함으로 인정
        return (intersection_area / box_area) >= threshold

    @staticmethod
    def calculate_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        """두 점(중심점 또는 박스 경계) 간의 유클리디안 거리 계산"""
        return float(np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))
