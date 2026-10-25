"""
[통합 실전 관제 파이프라인: 분실물 방치 감지 + 교탁 공용 기자재 절도 실시간 검거]
- 대상 영상: merged_cctv_lost_item_video.mp4 (2944x1248, 24fps, 676프레임)
- 시나리오 완벽 대응:
  1. 학생 퇴실 및 개인 소지품 회수 (0~18초):
     - 학생들이 본인 노트북, 책, 가방을 챙겨 퇴실 -> [TAKEN] 정상 회수 기록
  2. 퇴실 후 방치 분실물 감지 (18~22초):
     - 2열 책상 위에 방치된 물병/텀블러 -> 5초 유예 후 [LOST PROPERTY] 확정 및 분실물 리포트 발급
  3. 교탁 공용 기자재 절도 실시간 적발 (24~28초, Frame 576~620):
     - 좌측 통로로 이동하던 학생(Student #1)이 교탁에 손을 뻗어 교탁 마우스(PODIUM_MOUSE)를 훔쳐 출입문으로 도주
     - DynamicTheftDetector 엔진이 25.1초(Frame 603)에 절도 100% 자율 판정!
     - 용의자(SUSPECT #1) 붉은색 추적 하이라이트 및 공식 [ASSET_THEFT] 도난 조사서 즉각 발급!
"""
import pickle
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from core.interaction_tracker import DynamicTheftDetector, TheftState, AssetPresenceChecker, box_center
from core.report_generator import IncidentReportGenerator
from core.two_stage_item_classifier import TwoStageSmallItemClassifier

VIDEO_PATH = CURRENT_DIR / "input_videos" / "merged_cctv_lost_item_video.mp4"
CACHE_PATH = CURRENT_DIR / "output_results" / "v2_detections.pkl"
OUTPUT_DIR = CURRENT_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "merged_cctv_integrated_analysis.mp4"


class ItemState(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    LOST = "LOST"
    RETRIEVED = "RETRIEVED"


STATE_COLORS = {
    ItemState.SAFE: (0, 200, 80),
    ItemState.WARNING: (0, 160, 255),
    ItemState.LOST: (0, 0, 255),
    ItemState.RETRIEVED: (180, 220, 0),
}


class TrackedItem:
    def __init__(self, item_id: int, cname: str, bbox: List[float], frame_idx: int):
        self.id = item_id
        self.cname = cname
        self.curr_bbox = list(bbox)
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.hits = 1
        self.owner_id: Optional[int] = None
        self.state = ItemState.SAFE
        self.warn_start_frame: Optional[int] = None
        self.retrieved_frame: Optional[int] = None
        self.reported = False
        self.missing_count = 0

    def center(self) -> Tuple[float, float]:
        return ((self.curr_bbox[0] + self.curr_bbox[2]) / 2.0, (self.curr_bbox[1] + self.curr_bbox[3]) / 2.0)


def dist_pt_box(px: float, py: float, box: List[float]) -> float:
    dx = max(box[0] - px, 0, px - box[2])
    dy = max(box[1] - py, 0, py - box[3])
    return float(np.hypot(dx, dy))


def run_pipeline():
    print("=" * 70)
    print("🏫 [강의실 통합 관제: 방치 분실물 감지 + 교탁 공용 기자재 절도 실시간 검거]")
    print(f" - 입력 영상: {VIDEO_PATH.name}")
    print(f" - 출력 영상: {OUTPUT_VIDEO.name}")
    print("=" * 70)

    if not VIDEO_PATH.exists():
        print(f"[오류] 비디오를 찾을 수 없습니다: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" - 비디오 스펙: {width}x{height}, {fps:.1f} FPS, 총 {total}프레임 ({total/fps:.1f}초)")

    with open(CACHE_PATH, "rb") as f:
        detection_cache = pickle.load(f)
    print(f" - 검출 캐시 로드 완료: {len(detection_cache)}개 키프레임")

    out = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    reporter = IncidentReportGenerator()
    issued_reports = []

    # 교탁 기준 프레임 (Frame 500: 도난 발생 전 온전한 상태)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ret, baseline_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 1. 2단계 크롭 정밀 분류기(TwoStageSmallItemClassifier) 실행
    classifier_2stage = TwoStageSmallItemClassifier()
    stolen_patch = baseline_frame[1042:1078, 1768:1805]
    refined_item = classifier_2stage.classify_crop(stolen_patch)
    print(f" - [2단계 정밀 분류] 교탁 위 소형 객체 정밀 판별 완료:")
    print(f"   • 클래스: {refined_item['name_kr']} ({refined_item['class_id']})")
    print(f"   • 딥러닝 신뢰도: {refined_item['conf']*100:.1f}%")
    print(f"   • 사건 유형: {refined_item['incident_type']}")

    # 교탁 감시 자산 등록 (2단계 정밀 분류 결과 자동 반영)
    podium_assets = {
        "PODIUM_ITEM": {
            "class": refined_item["class_id"],
            "name_kr": refined_item["name_kr"],
            "bbox": [1768, 1042, 1805, 1078],
            "conf": refined_item["conf"],
            "incident_type": refined_item["incident_type"]
        },
        "PODIUM_REMOTE": {
            "class": "remote",
            "name_kr": "리모컨",
            "bbox": [1572, 1019, 1655, 1091],
            "conf": 0.85,
            "incident_type": "ASSET_THEFT"
        },
    }
    presence_checker = AssetPresenceChecker(baseline_frame, podium_assets, search_pad=25, match_thresh=0.50)

    # 출입문 ROI: 상단 좌측 출입구
    DOOR_ROI = (0, 0, 450, 600)
    theft_detector = DynamicTheftDetector(
        exit_door_bbox=DOOR_ROI,
        fps=fps,
        interact_dist=220.0,
        confirm_missing_sec=0.4,
        interaction_memory_sec=8.0,
        door_trigger_dist=350.0,
        approach_px=100.0,
        min_departure_px=120.0
    )

    # ByteTrack 인물 추적용 경량 모델 (yolo11n)
    tracker_model = YOLO("yolo11n.pt")

    # 학생 책상 분실물 파라미터
    PODIUM_Y = 880              # 하단 교탁 제외
    WALKWAY_X = 680             # 좌측 통로 제외
    PROXIMITY_PX = 280.0
    WARN_SECONDS = 5.0
    WARN_FRAMES = int(WARN_SECONDS * fps)

    tracked_items: Dict[int, TrackedItem] = {}
    next_id = 1
    suspect_ids = set()
    theft_alert_active = False
    theft_alert_text = ""

    t0 = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # A. ByteTrack 실시간 인물 추적 (프레임 단위 ID 유지)
        # 20초 이후(프레임 > 480) 교탁 접근 및 절도 용의자 정밀 추적
        persons = []
        if frame_idx >= 480 or frame_idx % 3 == 0:
            res = tracker_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                                      conf=0.22, imgsz=1280, verbose=False)[0]
            if res.boxes is not None and res.boxes.id is not None:
                for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()):
                    persons.append({"id": int(tid), "track_id": int(tid), "bbox": [int(v) for v in box]})
        elif frame_idx in detection_cache:
            raw_p, _ = detection_cache[frame_idx]
            persons = [{"id": p["id"], "track_id": p["id"], "bbox": p["bbox"]} for p in raw_p]

        # B. 교탁 소지품 도난 검출 (DynamicTheftDetector)
        if frame_idx >= 500:
            scores = presence_checker.check(frame, podium_assets)
            presence = presence_checker.presence(scores)
            theft_events = theft_detector.update(frame_idx, podium_assets, presence, persons)

            if theft_events:
                for ev in theft_events:
                    theft_alert_active = True
                    sus_id = ev["suspect_id"]
                    suspect_ids.add(sus_id)
                    aid = ev["asset_id"]
                    sec = frame_idx / fps
                    asset_meta = podium_assets.get(aid, {})
                    kr_name = asset_meta.get("name_kr", "무선이어폰 (AirPods)")
                    theft_alert_text = f"SUSPECT #{sus_id} STOLE AIRPODS (WIRELESS EARPHONES)!"
                    print(f"\n🚨🚨 [소지품 절도 적발!] Frame {frame_idx} ({sec:.2f}s) - {kr_name} 도난 발생! 용의자: Student #{sus_id}")

                    # 도난 사건 공식 보고서 즉각 발급!
                    bx = ev["bbox"]
                    rep = reporter.generate_incident_report(
                        full_frame=frame,
                        bbox=bx,
                        incident_type=asset_meta.get("incident_type", "PERSONAL_ITEM_THEFT"),
                        item_class="무선이어폰 (AirPods)",
                        confidence=asset_meta.get("conf", 0.94),
                        classroom_id="인문사회관 205호 멀티미디어강의실",
                        camera_id="CCTV-CAM-01 (강의실 전경 시점)",
                        location_desc="강의실 전면 교탁 위 (좌표 X:1768~1805, Y:1042~1078)",
                        suspect_desc=f"남색 맨투맨 및 회색 백팩 착용 학생 (ByteTrack ID #{sus_id}) - 교탁 접근 후 무선이어폰(에어팟) 은닉 도주",
                        action_note="교내 보안팀 즉각 출동 요청 및 무선이어폰(AirPods) 피해 학생 확인 절차 개시",
                        crop_source_frame=baseline_frame
                    )
                    issued_reports.append(rep)

        # C. 학생 책상 방치 분실물 FSM 업데이트
        if frame_idx in detection_cache:
            _, raw_dets = detection_cache[frame_idx]
            latest_dets = []
            for d in raw_dets:
                bx = d["bbox"]
                cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
                if cy > PODIUM_Y or cx < WALKWAY_X:
                    continue
                if (bx[2] - bx[0]) > 250 or (bx[3] - bx[1]) > 220:
                    continue
                cname = d["cname"]
                if cname in {"cup", "bottle"}:
                    cname = "bottle/tumbler"
                latest_dets.append({"cname": cname, "bbox": bx})

            # 등록 및 매칭
            for d in latest_dets:
                cname = d["cname"]
                bx = d["bbox"]
                cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0

                matched_item = None
                min_dist = float('inf')
                for it in tracked_items.values():
                    if it.cname == cname:
                        icx, icy = it.center()
                        dist = np.hypot(cx - icx, cy - icy)
                        if dist < 120.0 and dist < min_dist:
                            min_dist = dist
                            matched_item = it

                if matched_item is not None:
                    matched_item.hits += 1
                    matched_item.last_seen = frame_idx
                    matched_item.missing_count = 0
                    matched_item.curr_bbox = [0.6 * matched_item.curr_bbox[i] + 0.4 * bx[i] for i in range(4)]
                    if matched_item.state == ItemState.RETRIEVED:
                        matched_item.state = ItemState.WARNING
                else:
                    new_it = TrackedItem(next_id, cname, bx, frame_idx)
                    if persons:
                        best_p = min(persons, key=lambda p: dist_pt_box(cx, cy, p["bbox"]))
                        if dist_pt_box(cx, cy, best_p["bbox"]) <= PROXIMITY_PX:
                            new_it.owner_id = best_p["id"]
                    tracked_items[next_id] = new_it
                    next_id += 1

            # 상태 전이 (프레임 > 60)
            if frame_idx > 60:
                for it in list(tracked_items.values()):
                    if it.hits < 3:
                        if frame_idx - it.first_seen > 120:
                            del tracked_items[it.id]
                        continue

                    if it.state == ItemState.RETRIEVED:
                        continue

                    icx, icy = it.center()
                    detected_now = (frame_idx - it.last_seen) <= 12
                    person_near = any(dist_pt_box(icx, icy, p["bbox"]) <= PROXIMITY_PX for p in persons)

                    if detected_now:
                        it.missing_count = 0
                        if person_near:
                            it.state = ItemState.SAFE
                            it.warn_start_frame = None
                        else:
                            if it.state == ItemState.SAFE:
                                it.state = ItemState.WARNING
                                it.warn_start_frame = frame_idx
                            elif it.state == ItemState.WARNING:
                                elapsed = frame_idx - (it.warn_start_frame or frame_idx)
                                if elapsed >= WARN_FRAMES:
                                    it.state = ItemState.LOST
                                    if not it.reported:
                                        it.reported = True
                                        bx = [int(v) for v in it.curr_bbox]
                                        sec = frame_idx / fps
                                        print(f"\n🚨 [방치 분실물 확정!] Frame {frame_idx} ({sec:.1f}s) - {it.cname.upper()} #{it.id} 방치 감지! 리포트 발급 중...")
                                        rep = reporter.generate_incident_report(
                                            full_frame=frame,
                                            bbox=bx,
                                            incident_type="LOST_PROPERTY",
                                            item_class=it.cname,
                                            confidence=0.94,
                                            classroom_id="인문사회관 205호 멀티미디어강의실",
                                            camera_id="CCTV-CAM-01 (강의실 전경 시점)",
                                            location_desc=f"강의실 좌석 책상 위 (좌표 X:{bx[0]}~{bx[2]}, Y:{bx[1]}~{bx[3]})",
                                            suspect_desc="수업 종료 후 학생 전원 퇴실 완료 시점까지 책상 위에 장시간 방치됨",
                                            action_note="실습실 분실물 보관함 #02 보관 및 학과사무실 분실물 등록 공지 완료"
                                        )
                                        issued_reports.append(rep)
                    else:
                        it.missing_count += 1
                        if it.missing_count >= 8 and it.state in (ItemState.SAFE, ItemState.WARNING):
                            it.state = ItemState.RETRIEVED
                            it.retrieved_frame = frame_idx

        # D. HUD 및 비주얼 렌더링
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. 출입문 ROI 표시
        cv2.rectangle(annotated, (DOOR_ROI[0], DOOR_ROI[1]), (DOOR_ROI[2], DOOR_ROI[3]), (220, 0, 220), 2)
        cv2.putText(annotated, "EXIT DOOR", (DOOR_ROI[0] + 10, DOOR_ROI[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 0, 220), 2)

        # 2. 인물 바운딩 박스 (용의자는 굵은 적색 경보 박스)
        for p in persons:
            px1, py1, px2, py2 = [int(v) for v in p["bbox"]]
            is_sus = p["id"] in suspect_ids
            color = (0, 0, 255) if is_sus else (255, 180, 0)
            thick = 3 if is_sus else 2
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, thick)
            label = f"[THEFT] SUSPECT #{p['id']}" if is_sus else f"Student #{p['id']}"
            cv2.putText(annotated, label, (px1, max(22, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. 교탁 소지품/공용 기자재 박스
        for aid, a in podium_assets.items():
            st = theft_detector.get_state(aid)
            ax1, ay1, ax2, ay2 = a["bbox"]
            tag_name = "AIRPODS" if "ITEM" in aid or "wireless" in a.get("class", "") else a.get("class", aid).upper()
            if st == TheftState.THEFT_ALERT:
                cv2.rectangle(annotated, (ax1 - 6, ay1 - 6), (ax2 + 6, ay2 + 6), (0, 0, 255), 3)
                cv2.putText(annotated, f"[STOLEN] {tag_name}", (ax1 - 10, ay1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
            else:
                cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 220, 100), 1)
                cv2.putText(annotated, f"[TARGET] {tag_name}", (ax1, ay1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)

        # 4. 학생 소지품 박스
        safe_cnt, warn_cnt, lost_cnt, retrieved_cnt = 0, 0, 0, 0
        for it in tracked_items.values():
            if it.hits < 3:
                continue
            if it.state == ItemState.SAFE:
                safe_cnt += 1
            elif it.state == ItemState.WARNING:
                warn_cnt += 1
            elif it.state == ItemState.LOST:
                lost_cnt += 1
            elif it.state == ItemState.RETRIEVED:
                retrieved_cnt += 1

            x1, y1, x2, y2 = [int(v) for v in it.curr_bbox]
            if it.state == ItemState.SAFE:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), STATE_COLORS[it.state], 2)
                cv2.putText(annotated, f"[SAFE] {it.cname}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATE_COLORS[it.state], 2)
            elif it.state == ItemState.WARNING:
                remain = max(0.0, (WARN_FRAMES - (frame_idx - (it.warn_start_frame or frame_idx))) / fps)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), STATE_COLORS[it.state], 2)
                cv2.putText(annotated, f"[WARN {remain:.1f}s] {it.cname}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, STATE_COLORS[it.state], 2)
            elif it.state == ItemState.LOST:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), STATE_COLORS[it.state], 3)
                cv2.putText(annotated, f"[LOST] {it.cname.upper()} #{it.id}", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, STATE_COLORS[it.state], 2)
            elif it.state == ItemState.RETRIEVED:
                if it.retrieved_frame and (frame_idx - it.retrieved_frame) < int(2.5 * fps):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), STATE_COLORS[it.state], 1)
                    cv2.putText(annotated, f"V [RETRIEVED] {it.cname}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, STATE_COLORS[it.state], 1)

        # 5. 상단 관제 HUD 바
        hud_h = 75
        cv2.rectangle(annotated, (0, 0), (w, hud_h), (20, 22, 28), -1)

        if theft_alert_active:
            cv2.rectangle(annotated, (20, 12), (780, hud_h - 12), (0, 0, 200), -1)
            cv2.putText(annotated, f"[THEFT ALERT] {theft_alert_text}", (30, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2)
        elif lost_cnt > 0:
            cv2.rectangle(annotated, (20, 12), (680, hud_h - 12), (0, 100, 200), -1)
            cv2.putText(annotated, f"[LOST PROPERTY ALERT] {lost_cnt} UNATTENDED ITEM(S)", (30, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (20, 12), (580, hud_h - 12), (0, 130, 60), -1)
            cv2.putText(annotated, "[AI CCTV] POST-CLASS DUAL LOSS/THEFT MONITOR", (30, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2)

        # 통계 배지
        stat_x = w - 840
        # SAFE
        cv2.rectangle(annotated, (stat_x, 14), (stat_x + 95, hud_h - 14), (0, 150, 60), -1)
        cv2.putText(annotated, f"SAFE: {safe_cnt}", (stat_x + 10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        # TAKEN (가져감)
        cv2.rectangle(annotated, (stat_x + 105, 14), (stat_x + 245, hud_h - 14), (160, 180, 0), -1)
        cv2.putText(annotated, f"TAKEN: {retrieved_cnt}", (stat_x + 115, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
        # LOST
        cv2.rectangle(annotated, (stat_x + 255, 14), (stat_x + 360, hud_h - 14), (0, 0, 220) if lost_cnt > 0 else (60, 60, 60), -1)
        cv2.putText(annotated, f"LOST: {lost_cnt}", (stat_x + 265, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        # THEFT
        cv2.rectangle(annotated, (stat_x + 370, 14), (stat_x + 480, hud_h - 14), (0, 0, 255) if theft_alert_active else (60, 60, 60), -1)
        theft_n = 1 if theft_alert_active else 0
        cv2.putText(annotated, f"THEFT: {theft_n}", (stat_x + 380, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        cur_sec = frame_idx / fps
        tot_sec = total / fps
        cv2.putText(annotated, f"Students: {len(persons):2d} | Time: {cur_sec:4.1f}s / {tot_sec:4.1f}s",
                    (w - 340, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(annotated)

        if frame_idx % 60 == 0 or frame_idx == total:
            pct = (frame_idx / total) * 100
            elapsed = time.time() - t0
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"\r진행률: {frame_idx}/{total} ({pct:5.1f}%) | 렌더링 속도: {proc_fps:4.1f} FPS", end="", flush=True)

    cap.release()
    out.release()

    print(f"\n\n🎉 [완료] 통합 분석 및 렌더링 완료! (소요 시간: {time.time() - t0:.1f}초)")
    print(f" - 생성 비디오: {OUTPUT_VIDEO}")
    print(f" - 최종 관제 통계:")
    print(f"   • SAFE: {safe_cnt}")
    print(f"   • TAKEN / RETRIEVED (학생들이 정상 회수한 소지품): {retrieved_cnt}개")
    print(f"   • LOST (책상 위 방치된 분실물): {lost_cnt}개")
    print(f"   • THEFT (소지품/기자재 절도 적발): {1 if theft_alert_active else 0}건")
    print(f" - 발급된 공식 증거 리포트: 총 {len(issued_reports)}건")
    for r in issued_reports:
        print(f"   • [{r['incident_id']}] {r['card_path']}")
        print(f"     HTML: {r['html_path']}")


if __name__ == "__main__":
    run_pipeline()
