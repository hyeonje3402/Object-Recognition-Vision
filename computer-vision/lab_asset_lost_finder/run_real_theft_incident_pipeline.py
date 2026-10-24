"""
[동적 상호작용 기반 실전 도난 감지 + 증거 리포트 자동 발급 파이프라인]

- 프레임 번호 하드코딩 없음: 임의의 CCTV 영상에서 아래 조건이 시공간적으로 충족될 때만 경보 발령
    1) ByteTrack으로 추적 중인 사람이 등록 자산에 접촉
    2) 자산 위치에서 물품이 사라진 상태가 지속 (사람에 가려진 구간 제외)
    3) 접촉자가 자산에서 멀어지며 출입문 방향으로 이동
- 처리 순서
    1. 첫 프레임 YOLO 스캔 -> 골든 베이스라인 자산 등록 (모니터/키보드/마우스)
    2. 프레임마다 ByteTrack 사람 추적 + 기준 패치 템플릿 매칭으로 자산 존재 여부 확인
    3. DynamicTheftDetector 판정 -> THEFT_ALERT 시 IncidentReportGenerator 자동 발급

사용 예)
    python run_real_theft_incident_pipeline.py
    python run_real_theft_incident_pipeline.py --video input_videos/other.mp4 --door-roi 0.8 0 1 0.7
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from core.interaction_tracker import (AssetPresenceChecker, DynamicTheftDetector,
                                      TheftState)
from core.report_generator import IncidentReportGenerator

DEFAULT_VIDEO = CURRENT_DIR / "input_videos" / "Hailuo_Video_Use the uploaded reference ima_570081669656702981.mp4"
OUTPUT_DIR = CURRENT_DIR / "output_results"

# 출입문 ROI (프레임 대비 정규화 좌표 x1 y1 x2 y2) - 설치 강의실별로 1회 설정
DEFAULT_DOOR_ROI = (0.80, 0.0, 1.0, 0.70)

REPORT_CLASS_NAME = {"tv": "monitor"}
STATE_COLORS = {  # BGR
    TheftState.SAFE: (0, 220, 100),
    TheftState.INTERACTING: (0, 200, 255),
    TheftState.MISSING_SUSPECT: (0, 140, 255),
    TheftState.THEFT_ALERT: (0, 0, 255),
}


def parse_args():
    ap = argparse.ArgumentParser(description="동적 상호작용 기반 도난 감지 파이프라인")
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR / "real_theft_cctv_analyzed.mp4")
    ap.add_argument("--door-roi", type=float, nargs=4, default=DEFAULT_DOOR_ROI,
                    metavar=("X1", "Y1", "X2", "Y2"), help="출입문 ROI (정규화 좌표)")
    ap.add_argument("--asset-model", default=str(CURRENT_DIR / "yolov8x.pt"), help="베이스라인 등록용 정밀 모델")
    ap.add_argument("--person-model", default=str(CURRENT_DIR / "yolo11n.pt"), help="ByteTrack 사람 추적용 경량 모델")
    ap.add_argument("--classroom", default="IT-융합공학관 402호 멀티미디어실습실")
    ap.add_argument("--camera", default="CCTV-CAM-01 (통로/출입구 방면)")
    ap.add_argument("--max-frames", type=int, default=0, help="디버그용 최대 처리 프레임 (0=전체)")
    return ap.parse_args()


def register_baseline(frame, model):
    """첫 프레임에서 공용 자산(모니터/키보드/마우스)을 스캔하여 인벤토리 등록"""
    res = model.predict(frame, imgsz=1280, conf=0.15, verbose=False)[0]
    raw = []
    for b in res.boxes:
        cname = model.names[int(b.cls[0])]
        if cname in ("tv", "keyboard", "mouse"):
            raw.append((REPORT_CLASS_NAME.get(cname, cname),
                        [int(v) for v in b.xyxy[0].tolist()], float(b.conf[0])))
    raw.sort(key=lambda r: (r[1][1], r[1][0]))
    assets, counts = {}, {}
    for cname, bbox, conf in raw:
        counts[cname] = counts.get(cname, 0) + 1
        assets[f"{cname.upper()}_{counts[cname]:02d}"] = {"class": cname, "bbox": bbox, "conf": conf}
    return assets


def detect_tracked_persons(tracker_model, frame, imgsz=1280):
    """ByteTrack 기반 사람 추적 (프레임마다 호출, persist=True 로 ID 유지)"""
    res = tracker_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                              conf=0.25, imgsz=imgsz, verbose=False)[0]
    persons = []
    if res.boxes is not None and res.boxes.id is not None:
        for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()):
            persons.append({"track_id": int(tid), "bbox": [int(v) for v in box]})
    return persons


def draw_overlay(frame, assets, detector, persons, door_roi, suspect_ids, any_alert, report_ids,
                 frame_idx, total, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (door_roi[0], door_roi[1]), (door_roi[2], door_roi[3]), (200, 0, 200), 2)
    cv2.putText(frame, "EXIT DOOR ROI", (door_roi[0] + 8, door_roi[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        is_sus = p["track_id"] in suspect_ids
        color = (0, 0, 255) if is_sus else (255, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_sus else 2)
        label = f"SUSPECT #{p['track_id']}" if is_sus else f"Student #{p['track_id']}"
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    for aid, a in assets.items():
        st = detector.get_state(aid)
        x1, y1, x2, y2 = a["bbox"]
        if st in (TheftState.SAFE, TheftState.INTERACTING):
            cv2.rectangle(frame, (x1, y1), (x2, y2), STATE_COLORS[st], 1)
        else:
            cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), STATE_COLORS[st], 3)
            tag = "[STOLEN]" if st == TheftState.THEFT_ALERT else "[MISSING?]"
            cv2.putText(frame, f"{tag} {a['class'].upper()}", (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, STATE_COLORS[st], 2)

    hud_h = 75
    cv2.rectangle(frame, (0, 0), (w, hud_h), (20, 20, 24), -1)
    if any_alert:
        bg = (0, 0, 200) if (frame_idx // 10) % 2 == 0 else (0, 0, 130)
        cv2.rectangle(frame, (20, 12), (620, hud_h - 12), bg, -1)
        cv2.putText(frame, "[ALERT!] ASSET THEFT DETECTED", (35, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, "INCIDENT: " + ", ".join(report_ids), (640, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
    else:
        cv2.rectangle(frame, (20, 12), (520, hud_h - 12), (0, 150, 60), -1)
        cv2.putText(frame, "[SYSTEM ACTIVE] MONITORING ASSETS", (35, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Registered: {len(assets)}", (550, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"CCTV {frame_idx / fps:4.1f}s / {total / fps:4.1f}s", (w - 300, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)


def run_pipeline(args):
    print("=" * 70)
    print("[동적 도난 감지 파이프라인 시작]")
    print(f" - 입력 영상: {args.video.name}")
    print("=" * 70)
    if not args.video.exists():
        print(f"[오류] 영상을 찾을 수 없습니다: {args.video}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" - 비디오 스펙: {width}x{height}, {fps:.1f} FPS, {total}프레임")

    ret, first_frame = cap.read()
    if not ret:
        print("[오류] 첫 프레임을 읽을 수 없습니다.")
        return

    # 1. 골든 베이스라인 등록
    asset_model = YOLO(args.asset_model)
    assets = register_baseline(first_frame, asset_model)
    del asset_model
    print(f"[등록] 공용 자산 {len(assets)}개 (monitor/keyboard/mouse)")

    baseline_frame = first_frame.copy()
    checker = AssetPresenceChecker(baseline_frame, assets)

    rx1, ry1, rx2, ry2 = args.door_roi
    door_roi = (int(rx1 * width), int(ry1 * height), int(rx2 * width), int(ry2 * height))
    detector = DynamicTheftDetector(exit_door_bbox=door_roi, fps=fps)

    person_model = YOLO(args.person_model)
    reporter = IncidentReportGenerator()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx, any_alert = 0, False
    report_ids, reports = [], []
    suspect_ids = set()
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        persons = detect_tracked_persons(person_model, frame)
        scores = checker.check(frame, assets)
        presence = checker.presence(scores)
        events = detector.update(frame_idx, assets, presence, persons)

        for ev in events:
            any_alert = True
            suspect_ids.add(ev["suspect_id"])
            a = assets[ev["asset_id"]]
            sec = frame_idx / fps
            print(f"\n[도난 감지] frame {frame_idx} ({sec:.1f}s) {ev['asset_id']} "
                  f"suspect=#{ev['suspect_id']} door_dist={ev['door_distance']}px")
            x, y = [int(v) for v in ((ev["bbox"][0] + ev["bbox"][2]) / 2, (ev["bbox"][1] + ev["bbox"][3]) / 2)]
            ev_frame = frame.copy()
            sx1, sy1, sx2, sy2 = ev["suspect_bbox"]
            cv2.rectangle(ev_frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 4)
            rep = reporter.generate_incident_report(
                full_frame=ev_frame,
                crop_source_frame=baseline_frame,
                bbox=ev["bbox"],
                incident_type="ASSET_THEFT",
                item_class=a["class"],
                confidence=min(0.99, max(0.5, a.get("conf", 0.9))),
                classroom_id=args.classroom,
                camera_id=args.camera,
                location_desc=f"자산 {ev['asset_id']} (화면 좌표 {x},{y}) - 영상 {sec:.1f}초 시점",
                suspect_desc=(f"추적 ID #{ev['suspect_id']} 인물이 {ev['asset_id']}에 접촉한 뒤 해당 물품이 "
                              f"소멸, 출입문 방향으로 이동하여 반출한 것으로 판정"),
                action_note="보안 관제실 비상 알림 전송 및 해당 시점 고화질 CCTV 증거 영상 영구 보관",
            )
            report_ids.append(rep["incident_id"])
            reports.append(rep)

        annotated = frame.copy()
        draw_overlay(annotated, assets, detector, persons, door_roi, suspect_ids, any_alert,
                     report_ids, frame_idx, total, fps)
        out.write(annotated)

        if frame_idx % 30 == 0:
            print(f"\r진행 {frame_idx}/{total} ({frame_idx / total * 100:5.1f}%)", end="", flush=True)

    cap.release()
    out.release()
    print(f"\n\n[완료] {time.time() - t0:.0f}초 소요, 발급 리포트 {len(reports)}건")
    print(f" - 분석 비디오: {args.output}")
    for r in reports:
        print(f" - {r['incident_id']}: {r['card_path']}")
        print(f"   HTML: {r['html_path']}")
    if not reports:
        print(" - 도난 이벤트 없음 (door-roi/임계값 확인 필요)")


if __name__ == "__main__":
    run_pipeline(parse_args())
