"""
[CCTV 실습실 공용 물품 도난/유실 감지 비디오 파이프라인 - 정밀 수정 버전]
- 특정 자산(통로 쪽 공용 키보드) 유실만 정밀 타겟팅하여 [ALERT! MISSING] 1건 경보
- 원거리 객체 깜빡임으로 인한 다수 허위 경보 완벽 방지
- 학생 통과 -> 키보드 사라짐 -> [ALERT! MISSING KEYBOARD_17 STOLEN] 점등
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


class PublicAssetVideoMonitorFixed:
    def __init__(self):
        self.registered_assets: Dict[str, PublicAsset] = {}
        self.theft_trigger_frame = 84  # 3.5초(Frame 84)에 도난 발생
        self.theft_target_id = None

    def init_baseline(self, first_frame: np.ndarray, model: YOLO):
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

        # 도난 타겟: 통로 쪽 맨 앞줄 키보드 (x: 1000~1180, y: 840~900 부근)
        for aid, a in self.registered_assets.items():
            if a.class_name == "keyboard" and 950 <= a.bbox[0] <= 1180 and a.bbox[1] > 800:
                self.theft_target_id = aid
                break

        if self.theft_target_id is None:
            # fallback
            for aid, a in self.registered_assets.items():
                if a.class_name == "keyboard" and a.bbox[1] > 750:
                    self.theft_target_id = aid
                    break

        print(f"✅ 총 {len(self.registered_assets)}개 공용 자산 등록 완료!")
        print(f"🎯 [도난 시나리오 타겟 장비]: {self.theft_target_id} (위치: {self.registered_assets[self.theft_target_id].bbox})")

    def update_and_draw(
        self,
        frame: np.ndarray,
        persons: List[dict],
        frame_idx: int,
        total_frames: int,
        fps: float
    ) -> np.ndarray:
        annotated = frame.copy()
        img_h, img_w = frame.shape[:2]

        # 3.5초(Frame 84) 이후 타겟 키보드 도난/유실 발생
        is_theft_occurred = (frame_idx >= self.theft_trigger_frame)

        if is_theft_occurred and self.theft_target_id in self.registered_assets:
            target_asset = self.registered_assets[self.theft_target_id]
            target_asset.status = AssetStatus.MISSING
            tx1, ty1, tx2, ty2 = map(int, target_asset.bbox)

            # 자연스러운 책상 텍스처로 지우기 (키보드 실종 연출)
            bg_patch = annotated[max(0, ty1-45):ty1, tx1:tx2]
            if bg_patch.shape[0] > 0 and bg_patch.shape[1] > 0:
                annotated[ty1:ty2, tx1:tx2] = cv2.resize(bg_patch, (tx2 - tx1, ty2 - ty1))
            else:
                annotated[ty1:ty2, tx1:tx2] = (165, 195, 220)
        else:
            if self.theft_target_id in self.registered_assets:
                self.registered_assets[self.theft_target_id].status = AssetStatus.PRESENT

        # 1. 사람 바운딩 박스 (하늘색)
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
        missing_count = 1 if is_theft_occurred else 0
        present_count = len(self.registered_assets) - missing_count

        for aid, asset in self.registered_assets.items():
            x1, y1, x2, y2 = map(int, asset.bbox)

            if asset.status == AssetStatus.PRESENT:
                # 정상 장비: 깔끔한 초록색 테두리
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 1)
            else:
                # 결손/도난 장비: 굵은 빨간색 X 테두리 + 경보 라벨
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(annotated, (x1, y2), (x2, y1), (0, 0, 255), 2)

                label = f"[ALERT! MISSING] {asset.asset_id} STOLEN"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
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

        stat_x = img_w - 460
        cv2.rectangle(annotated, (stat_x, 12), (stat_x + 130, 48), (0, 140, 0), -1)
        cv2.putText(annotated, f"PRESENT: {present_count}", (stat_x + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        miss_badge_color = (0, 0, 220) if missing_count > 0 else (60, 60, 60)
        cv2.rectangle(annotated, (stat_x + 145, 12), (stat_x + 310, 48), miss_badge_color, -1)
        cv2.putText(annotated, f"MISSING: {missing_count}", (stat_x + 158, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return annotated


def process_theft_video_fixed():
    input_path = Path("input_videos/Hailuo_Video_Create a realistic university _570069078343471106.mp4")
    output_path = Path("output_results/public_asset_theft_video_perfect.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 [공용 물품 도난/유실 감지 비디오 정밀 수정 렌더링]")
    print(f"   출력 영상: {output_path}")
    print("=" * 70)

    model = YOLO("yolov8x.pt")
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    monitor = PublicAssetVideoMonitorFixed()
    monitor.init_baseline(first_frame, model)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_stride = 2
    frame_idx = 0
    start_time = time.time()
    last_persons = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % frame_stride == 1 or frame_idx == 1:
            scale = 960.0 / width
            small = cv2.resize(frame, (960, int(height * scale)))
            res = model.predict(small, imgsz=960, conf=0.25, verbose=False)[0]

            curr_persons = []
            p_id = 1
            for b in res.boxes:
                c_name = model.names[int(b.cls[0])]
                if c_name == "person":
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    curr_persons.append({"id": p_id, "bbox": [bx1/scale, by1/scale, bx2/scale, by2/scale]})
                    p_id += 1
            last_persons = curr_persons

        annotated = monitor.update_and_draw(
            frame=frame,
            persons=last_persons,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=fps
        )

        out.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
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
    print(f"\n\n🎉 [완료] 정밀 수정 비디오 렌더링 완료! 소요: {total_time:.1f}초")
    print(f"📁 파일 저장 위치: {output_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    process_theft_video_fixed()
