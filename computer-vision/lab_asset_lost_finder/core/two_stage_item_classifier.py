"""
[2단계 크롭 정밀 분류기 - core/two_stage_item_classifier.py]
YOLOv8 기본 COCO 80종 모델의 한계(무선이어폰/에어팟/버즈 클래스 부재로 인한 마우스 오분류)를 극복하기 위해,
검출된 소형 객체 크롭 패치(Bounding Box Crop)를 2차 심층 분석하여
'무선이어폰 (AirPods)' / '마우스 (Mouse)' / '리모컨 (Remote)' 등을 정밀 판별하는 2-Stage Classifier 모듈.
"""
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

class TwoStageSmallItemClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        try:
            self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=self.weights).eval()
            self.categories = self.weights.meta["categories"]
            self.transform = self.weights.transforms()
            self.has_model = True
        except Exception as e:
            print(f"[TwoStageSmallItemClassifier] DeepNet 초기화 경고: {e} (형상 분석 폴백 모드 사용)")
            self.has_model = False

        # ImageNet 카테고리 중 소형 케이스/에어팟 케이스 유사 라벨
        self.case_like_labels = {
            "face powder", "pill bottle", "lotion", "sunscreen",
            "soap dispenser", "packet", "carton", "modem", "ipod", "stopwatch"
        }
        self.mouse_labels = {"mouse, computer mouse", "trackball"}

    def classify_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        크롭 이미지(BGR)를 입력받아 정밀 클래스, 한국어 명칭, 신뢰도, 카테고리 및 사고 유형 반환
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {
                "class_id": "unknown",
                "name_kr": "미식별 물품",
                "conf": 0.5,
                "category": "UNKNOWN",
                "incident_type": "LOST_PROPERTY"
            }

        h, w = crop_bgr.shape[:2]
        aspect = max(w, h) / max(min(w, h), 1)

        # 1. 중앙 50% 영역 샘플링 (배경 책상 목재 노이즈 배제)
        cy, cx = h // 2, w // 2
        half_h, half_w = max(h // 4, 1), max(w // 4, 1)
        center_patch = crop_bgr[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
        hsv_c = cv2.cvtColor(center_patch, cv2.COLOR_BGR2HSV)
        sat_c = float(hsv_c[:, :, 1].mean())
        val_c = float(hsv_c[:, :, 2].mean())

        # 2. 딥러닝 특징 추출 (MobileNetV3)
        top_cats = []
        has_case_pred = False
        has_mouse_pred = False

        if self.has_model:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0)
            with torch.no_grad():
                probs = self.model(tensor).softmax(1)[0]
                top5_p, top5_c = torch.topk(probs, 5)
            top_cats = [self.categories[idx.item()] for idx in top5_c]
            has_case_pred = any(c in self.case_like_labels for c in top_cats)
            has_mouse_pred = any(c in self.mouse_labels for c in top_cats)

        # 3. 종합 다중 팩터 판별 규칙 (에어팟/무선이어폰 vs 마우스 vs 리모컨)
        is_bright_white = (val_c > 170 and sat_c < 50)
        is_squircle = (aspect < 1.38 and max(w, h) <= 65)

        if is_bright_white and is_squircle and not has_mouse_pred:
            # 고유 특성: 흰색/유광 조약돌형 소형 케이스 -> 무선이어폰 (AirPods) 확정
            conf = min(0.96, 0.70 + (val_c / 255.0) * 0.15 + (1.0 - sat_c / 255.0) * 0.10)
            return {
                "class_id": "wireless_earphones",
                "name_kr": "무선이어폰 (AirPods)",
                "conf": float(conf),
                "category": "PERSONAL_ITEM",
                "incident_type": "PERSONAL_ITEM_THEFT",
                "top_preds": top_cats[:3]
            }
        elif aspect > 2.1:
            # 길쭉한 바형 리모컨
            return {
                "class_id": "remote",
                "name_kr": "빔프로젝터 리모컨",
                "conf": 0.88,
                "category": "PUBLIC_ASSET",
                "incident_type": "ASSET_THEFT",
                "top_preds": top_cats[:3]
            }
        else:
            # 마우스
            return {
                "class_id": "mouse",
                "name_kr": "광학 마우스",
                "conf": 0.82,
                "category": "PUBLIC_ASSET",
                "incident_type": "ASSET_THEFT",
                "top_preds": top_cats[:3]
            }
