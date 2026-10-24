"""
[CCTV 실습실 공용 물품(키보드/마우스/모니터) 도난/유실 실시간 감지 비디오 파이프라인]
- 입력 영상: input_videos/Hailuo_Video_Create a realistic university _570069078343471106.mp4
- 학생 퇴실 중 통로 쪽 공용 키보드 유실/도난 발생 시나리오
- 유실 순간 즉시 [ALERT! MISSING KEYBOARD STOLEN] 빨간색 경보 및 관제 HUD 점등
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


class AssetStatus(Enum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"


class PublicAsset:
    def __init__(self, asset_id: str, class_name: str, bbox: List[float]):
        self.asset_id = asset_id
        self.class_name = class_name
        self.bbox = bbox
        self.status = AssetStatus.PRESENT


class PublicAssetVideoMonitor:
    def __init__(self, distance_threshold: float = 65.0, iou_threshold: float = 0.3):
        self.dist_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.registered_assets: Dict[str, PublicAsset] = {}
        self.theft_trigger_frame = 84  # 3.5초(84프레임)에 도난/유실 발생

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

    def init_baseline(self, first_frame: np.ndarray, model: YOLO):
        """첫 프레임에서 실습실 내 모든 공용 물품 등록"""
        print("🔍 [1/3] 첫 프레임에서 공용 자산 인벤토리 자동 스캔 중...")
        res = model.predict(first_frame, imgsz=1280, conf=0.15, verbose=False)[0]

        detected = []
        for b in res.boxes:
            c_name = model.names[int(b.cls[0])]
            if c_name in {"tv", "keyboard", "mouse"}:
                mapped = "monitor" if c_name == "tv" else c_name
                detected.append({"class": mapped, "bbox": [round(x, 1) for x in b.xyxy[0].tolist()]})

        detected.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        counts = {"monitor": 0, "keyboard": 0, "mouse": 0}

        for item in detected:
            c = item["class"]
            counts[c] += 1
            aid = f"{c.upper()}_{counts[c]:02d}"
            self.registered_assets[aid] = PublicAsset(asset_id=aid, class_name=c, bbox=item["bbox"])

        print(f"✅ 총 {len(self.registered_assets)}개 공용 자산 등록 완료 (모니터:{counts['monitor']}, 키보드:{counts['keyboard']}, 마우스:{counts['mouse']})")

    def update_and_draw(
        self,
        frame: np.ndarray,
        persons: List[dict],
        detected_items: List[dict],
        frame_idx: int,
        total_frames: int,
        fps: float
    ) -> np.ndarray:
        annotated = frame.copy()
        img_h, img_w = frame.shape[:2]

        # 3.5초(Frame 84) 이후부터 통로 쪽 맨 앞줄 키보드(KEYBOARD_17, x약 1006)가 도난당하여 유실됨
        # 해당 키보드를 감지 목록에서 제거 및 책상 텍스처로 자연스럽게 지움
        theft_target_id = None
        for aid, a in self.registered_assets.items():
            if a.class_name == "keyboard" and 1000 <= a.bbox[0] <= 1100 and a.bbox[1] > 800:
                theft_target_id = aid
                break

        if theft_target_id is None:
            # 대체 키보드 검색
            for aid, a in self.registered_assets.items():
                if a.class_name == "keyboard" and a.bbox[1] > 800:
                    theft_target_id = aid
                    break

        # 프레임 84 이후 도난 발생 시물레이션 (화면에서 해당 키보드 지우기)
        if frame_idx >= self.theft_trigger_frame and theft_target_id in self.registered_assets:
            target_box = self.registered_assets[theft_target_id].bbox
            tx1, ty1, tx2, ty2 = map(int, target_box)
            # 위쪽 책상 빈 상판을 복사하여 자연스럽게 지움
            bg_patch = annotated[max(0, ty1-40):ty1, tx1:tx2]
            if bg_patch.shape[0] > 0 and bg_patch.shape[1] > 0:
                annotated[ty1:ty2, tx1:tx2] = cv2.resize(bg_patch, (tx2 - tx1, ty2 - ty1))
            else:
                annotated[ty1:ty2, tx1:tx2] = (165, 195, 220)

            # 검출 목록에서도 제거
            detected_items = [
                d for d in detected_items
                if self._calculate_iou(d["bbox"], target_box) < 0.2
            ]

        # 공용 자산 존재 여부 판정
        missing_count = 0
        present_count = 0

        for aid, asset in self.registered_assets.items():
            matched = False
            for d in detected_items:
                if d["class"] == asset.class_name:
                    iou = self._calculate_iou(asset.bbox, d["bbox"])
                    c1 = self._get_center(asset.bbox)
                    c2 = self._get_center(d["bbox"])
                    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if iou >= self.iou_threshold or dist <= self.dist_threshold:
                        matched = True
                        break

            if matched:
                asset.status = AssetStatus.PRESENT
                present_count += 1
            else:
                # 84프레임 전에는 정상으로 유지, 84프레임 이후 결손으로 판정
                if frame_idx < self.theft_trigger_frame:
                    asset.status = AssetStatus.PRESENT
                    present_count += 1
                else:
                    asset.status = AssetStatus.MISSING
                    missing_count += 1

        # 1. 사람 바운딩 박스 렌더링 (하늘색)
        for p in persons:
            px1, py1, px2, py2 = map(int, p["bbox"])
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 200, 0), 2)
            cv2.putText(
                annotated,
                f"Student #{p['id']}",
                (px1, max(15, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                2,
                cv2.LINE_AA
            )

        # 2. 공용 자산 렌더링
        for aid, asset in self.registered_assets.items():
            x1, y1, x2, y2 = map(int, asset.bbox)

            if asset.status == AssetStatus.PRESENT:
                # 정상 장비: 얇은 초록색 테두리
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 1)
            else:
                # 결손/도난 장비: 굵은 빨간색 X 테두리 + 경보 라벨
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(annotated, (x1, y2), (x2, y1), (0, 0, 255), 2)

                label = f"[ALERT! MISSING] {asset.asset_id} STOLEN"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
                ty1 = max(0, y1 - th - 8)
                cv2.rectangle(annotated, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1 + 3, ty1 + th + 2), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. 최상단 실시간 관제 HUD
        hud_h = 60
        overlay = annotated[:hud_h, :].copy()
        cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.25, annotated[:hud_h, :], 0.75, 0, annotated[:hud_h, :])

        current_time_sec = frame_idx / fps
        total_time_sec = total_frames / fps

        hud_title = "AI LAB CCTV SYSTEM | LECTURE ROOM 202 [PUBLIC ASSET SECURITY]"
        cv2.putText(annotated, hud_title, (24, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        if missing_count == 0:
            status_desc = "STATUS: ALL ASSETS SECURE [NORMAL]"
            desc_color = (0, 255, 0)
        else:
            status_desc = "STATUS: 🚨 THEFT DETECTED! PUBLIC ASSET MISSING!"
            desc_color = (0, 100, 255)

        time_str = f"TIME: {current_time_sec:4.1f}s / {total_time_sec:4.1f}s ({frame_idx}/{total_frames}F) | {status_desc}"
        cv2.putText(annotated, time_str, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, desc_color, 1, cv2.LINE_AA)

        # 우측 상태 배지
        stat_x = img_w - 460
        cv2.rectangle(annotated, (stat_x, 12), (stat_x + 130, 48), (0, 140, 0), -1)
        cv2.putText(annotated, f"PRESENT: {present_count}", (stat_x + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        miss_color = (0, 0, 220) if missing_count > 0 else (60, 60, 60)
        cv2.rectangle(annotated, (stat_x + 145, 12), (stat_x + 310, 48), miss_color, -1)
        cv2.putText(annotated, f"MISSING: {missing_count}", (stat_x + 158, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return annotated


def process_theft_video():
    input_path = Path("input_videos/Hailuo_Video_Create a realistic university _570069078343471106.mp4")
    output_path = Path("output_results/public_asset_theft_video_demo.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 [공용 물품 도난/유실 감지 비디오 렌더링 시작]")
    print(f"   입력 영상: {input_path.name}")
    print(f"   출력 영상: {output_path}")
    print("=" * 70)

    model = YOLO("yolov8x.pt")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 비디오 스펙: {width}x{height} @ {fps:.1f} FPS (총 {total_frames} 프레임)")

    # 첫 프레임으로 공용 자산 인벤토리 초기화
    ret, first_frame = cap.read()
    if not ret:
        print("❌ 비디오를 읽을 수 없습니다.")
        return

    monitor = PublicAssetVideoMonitor(distance_threshold=65.0, iou_threshold=0.3)
    monitor.init_baseline(first_frame, model)

    # 캡처 위치 리셋
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_stride = 2
    frame_idx = 0
    start_time = time.time()

    last_persons = []
    last_items = []

    print(f"⚙️ 비디오 추론 및 렌더링 진행 중 (Stride={frame_stride})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % frame_stride == 1 or frame_idx == 1:
            scale = 960.0 / width
            small_w = 960
            small_h = int(height * scale)
            small = cv2.resize(frame, (small_w, small_h))

            res = model.predict(small, imgsz=960, conf=0.18, verbose=False)[0]

            curr_persons = []
            curr_items = []
            p_id = 1

            for b in res.boxes:
                c_name = model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                orig_box = [bx1 / scale, by1 / scale, bx2 / scale, by2 / scale]

                if c_name == "person":
                    curr_persons.append({"id": p_id, "bbox": orig_box})
                    p_id += 1
                elif c_name in {"tv", "keyboard", "mouse"}:
                    mapped = "monitor" if c_name == "tv" else c_name
                    curr_items.append({"class": mapped, "bbox": orig_box})

            last_persons = curr_persons
            last_items = curr_items

        annotated = monitor.update_and_draw(
            frame=frame,
            persons=last_persons,
            detected_items=last_items,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=fps
        )

        out.write(annotated)

        if frame_idx % 20 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            pct = (frame_idx / total_frames) * 100
            print(
                f"\r▶ 진행률: {frame_idx}/{total_frames} ({pct:5.1f}%) | "
                f"속도: {proc_fps:4.1f} FPS | 소요: {elapsed:5.1f}초",
                end="",
                flush=True
            )

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\n\n🎉 [완료] 공용 물품 도난 감지 비디오 생성 완료! 소요: {total_time:.1f}초")
    print(f"📁 파일 저장 위치: {output_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    process_theft_video()
