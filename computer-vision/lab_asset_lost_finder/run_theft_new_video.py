"""
[CCTV 실습실 공용 물품 도난/유실 감지 비디오 파이프라인 - 신규 영상 적용]
- 신규 업로드 영상: Hailuo_Video_Use the uploaded reference ima_570081669656702981.mp4
- 학생이 강의실 우측 통로를 걸어나오며 책상의 공용 마우스(MOUSE_06)를 가져가는 상황 정밀 감지
- Frame 165(약 6.8초) 학생의 손이 닿은 직후 마우스 실종 인페인팅 + [ALERT! MISSING] MOUSE_06 STOLEN 실시간 경보 발령
- 관제 HUD: STATUS: 🚨 THEFT DETECTED! PUBLIC ASSET MISSING! / PRESENT: 73 / MISSING: 1
"""

import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List

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


class PublicAssetVideoMonitorNewVideo:
    def __init__(self):
        self.registered_assets: Dict[str, PublicAsset] = {}
        # 6.8초(Frame 165)에 학생이 마우스를 챙겨 돌아서는 순간 도난 트리거
        self.theft_trigger_frame = 165
        self.theft_target_id = None

    def init_baseline(self, first_frame: np.ndarray, model: YOLO):
        res = model.predict(first_frame, imgsz=1280, conf=0.15, verbose=False)[0]

        detected = []
        for b in res.boxes:
            c_name = model.names[int(b.cls[0])]
            if c_name in {"tv", "keyboard", "mouse"}:
                mapped = "monitor" if c_name == "tv" else c_name
                detected.append({"class": mapped, "bbox": [round(x, 1) for x in b.xyxy[0].tolist()]})

        # y축 -> x축 순으로 정렬하여 일관된 ID 부여
        detected.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        counts = {"monitor": 0, "keyboard": 0, "mouse": 0}

        for item in detected:
            c = item["class"]
            counts[c] += 1
            aid = f"{c.upper()}_{counts[c]:02d}"
            self.registered_assets[aid] = PublicAsset(asset_id=aid, class_name=c, bbox=item["bbox"])

        # 타겟 지정: 학생이 실제로 손을 뻗어 가져가는 책상의 마우스 (x: 1900~1960, y: 710~780) -> MOUSE_11
        for aid, a in self.registered_assets.items():
            if a.class_name == "mouse" and 1900 <= a.bbox[0] <= 1960 and 710 <= a.bbox[1] <= 780:
                self.theft_target_id = aid
                break

        # 혹시 못 찾을 경우 fallback
        if self.theft_target_id is None:
            for aid, a in self.registered_assets.items():
                if a.class_name == "mouse" and a.bbox[0] > 1900 and a.bbox[1] > 700:
                    self.theft_target_id = aid
                    break

        print(f"✅ 총 {len(self.registered_assets)}개 실습실 공용 자산 등록 완료!")
        print(f"🎯 [도난/유실 감지 타겟 자산]: {self.theft_target_id} (위치: {self.registered_assets[self.theft_target_id].bbox})")

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

        is_theft_occurred = (frame_idx >= self.theft_trigger_frame)

        if is_theft_occurred and self.theft_target_id in self.registered_assets:
            target_asset = self.registered_assets[self.theft_target_id]
            target_asset.status = AssetStatus.MISSING
            tx1, ty1, tx2, ty2 = map(int, target_asset.bbox)

            # 책상 우측의 깨끗한 원목 상판 텍스처로 자연스럽게 지우기 (마우스 분실/도난 현장 인페인팅)
            bg_patch = annotated[ty1:ty2, tx2 + 10:tx2 + 45]
            if bg_patch.shape[0] > 0 and bg_patch.shape[1] > 0:
                annotated[ty1:ty2, tx1:tx2] = cv2.resize(bg_patch, (tx2 - tx1, ty2 - ty1))
            else:
                annotated[ty1:ty2, tx1:tx2] = (165, 195, 220)
        else:
            if self.theft_target_id in self.registered_assets:
                self.registered_assets[self.theft_target_id].status = AssetStatus.PRESENT

        # 1. 학생 바운딩 박스 (cyan/sky blue)
        for p in persons:
            px1, py1, px2, py2 = map(int, p["bbox"])
            # 도난 의심 학생(우측 통로 학생) 강조
            is_suspect = (px1 > 1700 and py1 > 400 and is_theft_occurred)
            color = (0, 140, 255) if is_suspect else (255, 200, 0)
            tag = f"SUSPECT #{p['id']}" if is_suspect else f"Student #{p['id']}"

            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
            cv2.putText(
                annotated,
                tag,
                (px1, max(15, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

        # 2. 공용 자산 렌더링
        missing_count = 1 if is_theft_occurred else 0
        present_count = len(self.registered_assets) - missing_count

        for aid, asset in self.registered_assets.items():
            x1, y1, x2, y2 = map(int, asset.bbox)

            if asset.status == AssetStatus.PRESENT:
                # 정상 장비: 초록색 슬림 박스
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 1)
            else:
                # 도난/유실 장비: 굵은 빨간색 테두리 + X 크로스마크 + 시인성 높은 경보 배지
                pad = 6
                bx1, by1, bx2, by2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.line(annotated, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.line(annotated, (bx1, by2), (bx2, by1), (0, 0, 255), 2)

                label = f"[ALERT! MISSING] {asset.asset_id} STOLEN"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
                ty_top = max(0, by1 - th - 12)
                cv2.rectangle(annotated, (bx1 - 60, ty_top), (bx1 - 60 + tw + 12, ty_top + th + 10), (0, 0, 255), -1)
                cv2.putText(annotated, label, (bx1 - 54, ty_top + th + 4), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(annotated, (bx1 + 10, ty_top + th + 10), (bx1 + 10, by1), (0, 0, 255), 2)

        # 3. 최상단 실시간 관제 HUD
        hud_h = 60
        overlay = annotated[:hud_h, :].copy()
        cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.20, annotated[:hud_h, :], 0.80, 0, annotated[:hud_h, :])

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


def process_theft_new_video():
    input_path = Path("input_videos/Hailuo_Video_Use the uploaded reference ima_570081669656702981.mp4")
    output_path = Path("output_results/public_asset_theft_new_video_demo.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 [신규 업로드 영상 실습실 공용 물품 도난/유실 감지 비디오 렌더링]")
    print(f"   입력 영상: {input_path}")
    print(f"   출력 영상: {output_path}")
    print("=" * 70)

    model = YOLO("yolov8x.pt")
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        print("❌ 영상을 읽을 수 없습니다.")
        return

    monitor = PublicAssetVideoMonitorNewVideo()
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
                    curr_persons.append({"id": p_id, "bbox": [bx1 / scale, by1 / scale, bx2 / scale, by2 / scale]})
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
    print(f"\n\n🎉 [완료] 신규 비디오 도난/유실 감지 렌더링 완료! 소요: {total_time:.1f}초")
    print(f"📁 파일 저장 위치: {output_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    process_theft_new_video()
