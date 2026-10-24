"""
[강의실/실습실 공용 물품(고정 자산) 등록 및 도난/유실(MISSING) 실시간 감지 시스템]
- 기준 사진(Golden Baseline)을 분석하여 실습실 내 모든 공용 물품(모니터, 키보드, 마우스 등) 자동 등록
- 새로운 프레임/사진 입력 시 등록된 위치의 물품 존재 여부를 전수 검사
- 결손/도난 발생 시 [ALERT! ASSET MISSING] 빨간색 경보 및 관제 HUD 표시
"""

import json
import math
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class AssetStatus(Enum):
    PRESENT = "PRESENT"    # 등록 위치에 정상 거치된 상태 (초록색)
    MISSING = "MISSING"    # 등록 위치에서 유실/도난된 결손 상태 (빨간색 경보)


class PublicAsset:
    def __init__(self, asset_id: str, class_name: str, bbox: List[float]):
        self.asset_id = asset_id
        self.class_name = class_name
        self.bbox = bbox  # 기준 위치 [x1, y1, x2, y2]
        self.status = AssetStatus.PRESENT
        self.last_detected_time = time.time()


class PublicAssetTracker:
    def __init__(self, iou_threshold: float = 0.35, dist_threshold: float = 60.0):
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.registered_assets: Dict[str, PublicAsset] = {}
        self.model = YOLO("yolov8x.pt")

    @staticmethod
    def _calculate_iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    @staticmethod
    def _get_center(box: List[float]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def register_baseline(self, baseline_img: np.ndarray, save_json_path: Optional[Path] = None):
        """
        기준 이미지에서 모든 공용 물품을 검출하여 고유 자산 번호로 등록
        """
        print("🔍 [1/3] 기준 이미지에서 공용 물품(모니터, 키보드, 마우스) 자동 스캔 중...")
        res = self.model.predict(baseline_img, imgsz=1280, conf=0.15, verbose=False)[0]

        counts = {"monitor": 0, "keyboard": 0, "mouse": 0}
        self.registered_assets.clear()

        # 화면 상단(y) -> 하단 순서로 정렬하여 번호 부여
        detected_raw = []
        for b in res.boxes:
            c_name = self.model.names[int(b.cls[0])]
            if c_name in {"tv", "keyboard", "mouse"}:
                mapped_name = "monitor" if c_name == "tv" else c_name
                detected_raw.append({
                    "class": mapped_name,
                    "bbox": [round(x, 1) for x in b.xyxy[0].tolist()]
                })

        # y좌표 기준 정렬 (책상 줄 번호 순서)
        detected_raw.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

        for item in detected_raw:
            c_name = item["class"]
            counts[c_name] += 1
            asset_id = f"{c_name.upper()}_{counts[c_name]:02d}"

            asset = PublicAsset(asset_id=asset_id, class_name=c_name, bbox=item["bbox"])
            self.registered_assets[asset_id] = asset

        print(f"✅ [등록 완료] 총 {len(self.registered_assets)}개 공용 자산 인벤토리 등록 완료!")
        print(f"   • 모니터: {counts['monitor']}대 | 키보드: {counts['keyboard']}개 | 마우스: {counts['mouse']}개")

        if save_json_path:
            save_json_path.parent.mkdir(parents=True, exist_ok=True)
            export_data = {
                aid: {"class": a.class_name, "bbox": a.bbox}
                for aid, a in self.registered_assets.items()
            }
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"💾 자산 DB 저장 위치: {save_json_path}")

    def audit_frame(self, current_img: np.ndarray) -> Dict[str, AssetStatus]:
        """
        현재 프레임을 검사하여 등록된 공용 자산의 결손(MISSING) 여부 판정
        """
        res = self.model.predict(current_img, imgsz=1280, conf=0.15, verbose=False)[0]

        current_items = []
        for b in res.boxes:
            c_name = self.model.names[int(b.cls[0])]
            if c_name in {"tv", "keyboard", "mouse"}:
                mapped_name = "monitor" if c_name == "tv" else c_name
                current_items.append({
                    "class": mapped_name,
                    "bbox": b.xyxy[0].tolist()
                })

        # 등록된 각 공용 자산별로 현재 검출된 객체와 매칭
        for asset_id, asset in self.registered_assets.items():
            matched = False
            for cur in current_items:
                if cur["class"] == asset.class_name:
                    iou = self._calculate_iou(asset.bbox, cur["bbox"])
                    c1 = self._get_center(asset.bbox)
                    c2 = self._get_center(cur["bbox"])
                    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

                    # IoU가 0.3 이상이거나 중심점 거리가 50px 이내면 동일 자산 거치 중으로 판정
                    if iou >= self.iou_threshold or dist <= self.dist_threshold:
                        matched = True
                        break

            if matched:
                asset.status = AssetStatus.PRESENT
            else:
                asset.status = AssetStatus.MISSING

        return {aid: a.status for aid, a in self.registered_assets.items()}

    def draw_audit_view(self, frame: np.ndarray) -> np.ndarray:
        """
        현재 감사 결과를 화면에 렌더링 (정상: 초록 Bbox, 결손: 빨간색 점선/굵은 Bbox + 결손 경보 알람)
        """
        annotated = frame.copy()
        img_h, img_w = frame.shape[:2]

        missing_count = 0
        present_count = 0

        for asset_id, asset in self.registered_assets.items():
            x1, y1, x2, y2 = map(int, asset.bbox)

            if asset.status == AssetStatus.PRESENT:
                present_count += 1
                color = (0, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            else:
                missing_count += 1
                # 결손(MISSING): 굵은 빨간색 경보 테두리
                color = (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                # 경보 텍스트 태그
                label = f"[ALERT! MISSING] {asset.asset_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
                ty1 = max(0, y1 - th - 8)
                cv2.rectangle(annotated, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1 + 3, ty1 + th + 2), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

                # 중앙 X 표시
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(annotated, (x1, y2), (x2, y1), (0, 0, 255), 2)

        # 상단 관제 HUD
        hud_h = 60
        overlay = annotated[:hud_h, :].copy()
        cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.25, annotated[:hud_h, :], 0.75, 0, annotated[:hud_h, :])

        title = "LAB ASSET AUDIT SYSTEM | LECTURE ROOM 202 [PUBLIC ASSET MONITOR]"
        cv2.putText(annotated, title, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        stat_info = f"TOTAL ASSETS: {len(self.registered_assets)} | STATUS: {'ALL SECURE' if missing_count == 0 else '🚨 THEFT / ASSET MISSING DETECTED!'}"
        info_color = (0, 255, 0) if missing_count == 0 else (0, 100, 255)
        cv2.putText(annotated, stat_info, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1, cv2.LINE_AA)

        # 우측 상태 배지
        stat_x = img_w - 450
        cv2.rectangle(annotated, (stat_x, 12), (stat_x + 130, 48), (0, 140, 0), -1)
        cv2.putText(annotated, f"PRESENT: {present_count}", (stat_x + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        miss_color = (0, 0, 220) if missing_count > 0 else (60, 60, 60)
        cv2.rectangle(annotated, (stat_x + 145, 12), (stat_x + 300, 48), miss_color, -1)
        cv2.putText(annotated, f"MISSING: {missing_count}", (stat_x + 158, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return annotated


def run_demo():
    input_img_path = Path("input_images/cb0908b8-d89e-49f6-b8e6-55cdf68d46fb.png")
    out_dir = Path("output_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_frame = cv2.imread(str(input_img_path))
    if base_frame is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {input_img_path}")
        return

    tracker = PublicAssetTracker()
    tracker.register_baseline(base_frame, save_json_path=out_dir / "lab_assets_inventory.json")

    # [시나리오 1]: 정상 상태 검사 (결손 0개)
    print("\n▶ [시나리오 1] 기준 상태 감사 수행 (MISSING: 0 기대)...")
    tracker.audit_frame(base_frame)
    safe_view = tracker.draw_audit_view(base_frame)
    cv2.imwrite(str(out_dir / "public_asset_baseline_ok.jpg"), safe_view)
    print(f"📸 정상 상태 뷰 저장: {out_dir / 'public_asset_baseline_ok.jpg'}")

    # [시나리오 2]: 맨 앞자리 2번 좌석의 모니터, 키보드, 마우스가 사라진 결손 상황 시뮬레이션
    print("\n▶ [시나리오 2] 맨 앞자리 장비(모니터, 키보드, 마우스) 도난/유실 결손 시뮬레이션...")
    theft_sim_frame = base_frame.copy()

    # 맨 앞줄 좌석 중 하나의 모니터/키보드/마우스 위치를 책상 색상으로 인페인팅(제거)
    # 앞줄 4개 좌석 중 우측 3번째 좌석 (x: 약 780~1080, y: 약 650~1050)
    # 모니터: [770, 650, 920, 800], 키보드: [740, 800, 920, 840], 마우스: [930, 800, 970, 840]
    # 사실적인 제거를 위해 책상 상판 텍스처 복사
    desk_color = [180, 200, 220]  # 책상 톤
    # 맨 앞줄 모니터 1대와 키보드 1개, 마우스 1개 영역을 책상 바닥으로 덮어씀 (장비 사라짐)
    # 실제 등록된 자산 중 맨 앞줄 자산 찾기
    front_monitor = [a for a in tracker.registered_assets.values() if a.class_name == "monitor" and a.bbox[1] > 600][2]
    front_keyboard = [a for a in tracker.registered_assets.values() if a.class_name == "keyboard" and a.bbox[1] > 700][2]
    front_mouse = [a for a in tracker.registered_assets.values() if a.class_name == "mouse" and a.bbox[1] > 700][2]

    target_missing_assets = [front_monitor, front_keyboard, front_mouse]
    for target in target_missing_assets:
        x1, y1, x2, y2 = map(int, target.bbox)
        # 주변 책상 영역으로 자연스럽게 블러/패치
        patch = theft_sim_frame[max(0, y1-30):y1, x1:x2]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            patch_resized = cv2.resize(patch, (x2 - x1, y2 - y1))
            theft_sim_frame[y1:y2, x1:x2] = patch_resized
        else:
            theft_sim_frame[y1:y2, x1:x2] = (160, 190, 215)

    # 결손 감사 수행
    tracker.audit_frame(theft_sim_frame)
    alert_view = tracker.draw_audit_view(theft_sim_frame)
    cv2.imwrite(str(out_dir / "public_asset_missing_alert.jpg"), alert_view)
    print(f"🚨 [경보 발생!] 결손 감지 결과 뷰 저장: {out_dir / 'public_asset_missing_alert.jpg'}")

    # [시나리오 3]: 시연용 동영상 생성 (정상 상태 3초 -> 장비 결손 발생 5초 -> 경보 HUD 점등)
    print("\n▶ [시나리오 3] 발표 및 시연용 영상(public_asset_missing_demo.mp4) 렌더링 중...")
    vid_out_path = out_dir / "public_asset_missing_demo.mp4"
    h, w = base_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24.0
    out_vid = cv2.VideoWriter(str(vid_out_path), fourcc, fps, (w, h))

    # 0~3초: 정상 상태 유지 (72 프레임)
    for _ in range(int(fps * 3)):
        out_vid.write(safe_view)

    # 3~8초: 장비 유실 발생 및 빨간색 MISSING 경보 점등 (120 프레임)
    for _ in range(int(fps * 5)):
        out_vid.write(alert_view)

    out_vid.release()
    print(f"🎉 시연 영상 생성 완료: {vid_out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
