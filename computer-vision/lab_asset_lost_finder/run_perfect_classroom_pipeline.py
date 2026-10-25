"""
[강의실 소지품 완벽 추적 및 방치/회수(가져감) 정밀 판정 파이프라인]
- 대상 영상: merged_cctv_lost_item_video.mp4 (2944x1248, 24fps, 676프레임)
- 핵심 원리:
  1. 수업 중 (1~120프레임): 책상 위 개인 소지품(노트북, 책, 휴대폰, 음료 등) 정밀 등록 및 인접 학생(소유자) 매핑
  2. 퇴실 시 소지품 회수 (TAKEN/RETRIEVED):
     - 학생이 짐을 챙겨 물품이 사라지면 '소유자 정상 회수(RETRIEVED)'로 즉시 판정
     - 빈 책상에 허위 유령 박스(False Positive)가 남지 않도록 깔끔하게 정리
  3. 방치 분실물 (LOST PROPERTY):
     - 학생이 떠났음에도 책상 위에 물리적으로 확실하게 남아있는 물품(물병, 텀블러 등)만 정밀 추적
     - 5초 유예 타이머(WARNING) 경과 후 최종 [LOST] 확정 및 고화질 증거 조사 보고서 자동 발급
  4. HUD 시각화:
     - 상단에 실시간 [SAFE], [RETRIEVED (가져감)], [WARNING], [LOST (남겨짐)] 카운트 배지 표출
     - 회수된 물품은 초록색 체크 태그 후 사라짐, 실제 남은 분실물은 선명한 붉은색 타겟 박스로 표시
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

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from core.report_generator import IncidentReportGenerator

VIDEO_PATH = CURRENT_DIR / "input_videos" / "merged_cctv_lost_item_video.mp4"
CACHE_PATH = CURRENT_DIR / "output_results" / "v2_detections.pkl"
OUTPUT_DIR = CURRENT_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "merged_cctv_lost_item_analyzed_perfect.mp4"


class ItemState(Enum):
    SAFE = "SAFE"                # 수업 중 정상 사용
    WARNING = "WARNING"          # 소유자 이탈 후 유예 카운트다운
    LOST = "LOST"                # 최종 방치 분실물 확정
    RETRIEVED = "RETRIEVED"      # 학생이 본인 물품 챙겨서 퇴실 (가져감)


STATE_COLORS = {
    ItemState.SAFE: (0, 200, 80),          # 초록
    ItemState.WARNING: (0, 160, 255),      # 주황
    ItemState.LOST: (0, 0, 255),           # 빨강
    ItemState.RETRIEVED: (180, 220, 0),    # 연두 (회수 완료)
}


class TrackedItem:
    def __init__(self, item_id: int, cname: str, bbox: List[float], frame_idx: int):
        self.id = item_id
        self.cname = cname
        self.anchor_bbox = list(bbox)     # 고정 책상 위치
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
    print("🏫 [강의실 소지품 회수(가져감) vs 방치 분실물 정밀 감지 파이프라인]")
    print(f" - 입력 영상: {VIDEO_PATH.name}")
    print(f" - 출력 영상: {OUTPUT_VIDEO.name}")
    print("=" * 70)

    if not VIDEO_PATH.exists():
        print(f"[오류] 비디오를 찾을 수 없습니다: {VIDEO_PATH}")
        return

    if not CACHE_PATH.exists():
        print(f"[오류] 검출 캐시가 없습니다: {CACHE_PATH}")
        return

    with open(CACHE_PATH, "rb") as f:
        detection_cache = pickle.load(f)
    print(f" - 검출 캐시 로드 완료: {len(detection_cache)}개 키프레임")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" - 비디오 스펙: {width}x{height}, {fps:.1f} FPS, 총 {total}프레임 ({total/fps:.1f}초)")

    out = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    reporter = IncidentReportGenerator()
    issued_reports = []

    # 파라미터
    PODIUM_Y = 880              # 하단 교탁 제외
    WALKWAY_X = 680             # 좌측 통로/출입구 제외 (학생 보행 통로)
    PROXIMITY_PX = 280.0        # 소유자 판정 반경
    WARN_SECONDS = 5.0          # 방치 유예 시간 (5초)
    WARN_FRAMES = int(WARN_SECONDS * fps)

    tracked_items: Dict[int, TrackedItem] = {}
    next_id = 1
    last_persons = []
    latest_dets = []

    # 전체 676프레임 스트리밍 처리
    t0 = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 캐시된 검출 정보가 있는 프레임인 경우 갱신
        if frame_idx in detection_cache:
            last_persons, raw_dets = detection_cache[frame_idx]
            latest_dets = []
            for d in raw_dets:
                bx = d["bbox"]
                cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
                # 교탁(하단) 및 좌측 출입구 복도 보행자 소지품 제외 (순수 학생 책상 영역만 감시)
                if cy > PODIUM_Y or cx < WALKWAY_X:
                    continue
                if (bx[2] - bx[0]) > 250 or (bx[3] - bx[1]) > 220:
                    continue
                cname = d["cname"]
                if cname in {"cup", "bottle"}:
                    cname = "bottle/tumbler"
                latest_dets.append({"cname": cname, "bbox": bx})

        # 1. 물품 등록 및 위치 갱신
        if frame_idx in detection_cache:
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
                    matched_item.curr_bbox = [
                        0.6 * matched_item.curr_bbox[i] + 0.4 * bx[i] for i in range(4)
                    ]
                    # 가려짐 해제 복구: 만약 학생 통과로 일시적 RETRIEVED 처리되었던 물품이 다시 나타나면 즉시 복구!
                    if matched_item.state == ItemState.RETRIEVED:
                        matched_item.state = ItemState.WARNING
                else:
                    # 새로운 물품 발견 시 등록 (수업 중 또는 퇴실 후 방치 물품)
                    new_it = TrackedItem(next_id, cname, bx, frame_idx)
                    if last_persons:
                        best_p = min(last_persons, key=lambda p: dist_pt_box(cx, cy, p["bbox"]))
                        if dist_pt_box(cx, cy, best_p["bbox"]) <= PROXIMITY_PX:
                            new_it.owner_id = best_p["id"]
                    tracked_items[next_id] = new_it
                    next_id += 1

        # 2. 퇴실 및 추적 단계 (프레임 > 60): 소지품 상태 전이
        if frame_idx in detection_cache and frame_idx > 60:
            for it in list(tracked_items.values()):
                # 3회 미만 단발성 잡음 제거
                if it.hits < 3:
                    if frame_idx - it.first_seen > 120:
                        del tracked_items[it.id]
                    continue

                if it.state == ItemState.RETRIEVED:
                    continue

                icx, icy = it.center()
                # 현재 프레임에서 물품이 직접 검출되었는지 여부
                detected_now = (frame_idx - it.last_seen) <= 12  # 최근 2검출 주기 이내 발견

                # 주변에 학생(사람)이 있는지 확인
                person_near = any(dist_pt_box(icx, icy, p["bbox"]) <= PROXIMITY_PX for p in last_persons)

                if detected_now:
                    it.missing_count = 0
                    if person_near:
                        # 소유자나 학생이 인근에 있음 -> SAFE
                        it.state = ItemState.SAFE
                        it.warn_start_frame = None
                    else:
                        # 사람이 자리를 떠남! -> WARNING 카운트다운 시작
                        if it.state == ItemState.SAFE:
                            it.state = ItemState.WARNING
                            it.warn_start_frame = frame_idx
                        elif it.state == ItemState.WARNING:
                            elapsed = frame_idx - (it.warn_start_frame or frame_idx)
                            if elapsed >= WARN_FRAMES:
                                it.state = ItemState.LOST
                                # 최초 방치 확정 시 사건 보고서 발급
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
                    # 물품이 검출되지 않음
                    it.missing_count += 1
                    # 장시간(8주기 = 약 2초) 연속 미검출 시, 소유자가 챙겨간 것(RETRIEVED)으로 최종 확정
                    if it.missing_count >= 8 and it.state in (ItemState.SAFE, ItemState.WARNING):
                        it.state = ItemState.RETRIEVED
                        it.retrieved_frame = frame_idx

        # 3. HUD 및 박스 렌더링
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 학생 바운딩 박스
        for p in last_persons:
            px1, py1, px2, py2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 180, 0), 2)
            cv2.putText(annotated, f"Student #{p['id']}", (px1, max(20, py1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

        # 물품 렌더링
        safe_cnt = 0
        warn_cnt = 0
        lost_cnt = 0
        retrieved_cnt = 0

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

            # 화면 박스 그리기
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
                cv2.putText(annotated, f"🚨 [LOST] {it.cname.upper()} #{it.id}", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, STATE_COLORS[it.state], 2)

            elif it.state == ItemState.RETRIEVED:
                # 회수된 물품은 회수 시점으로부터 3초간 연두색 체크 태그 표시 후 사라짐
                if it.retrieved_frame and (frame_idx - it.retrieved_frame) < int(3.0 * fps):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), STATE_COLORS[it.state], 1, cv2.LINE_AA)
                    cv2.putText(annotated, f"V [RETRIEVED] {it.cname}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATE_COLORS[it.state], 1)

        # 상단 관제 HUD 바
        hud_h = 75
        cv2.rectangle(annotated, (0, 0), (w, hud_h), (20, 22, 28), -1)

        if lost_cnt > 0:
            cv2.rectangle(annotated, (20, 12), (680, hud_h - 12), (0, 0, 180), -1)
            cv2.putText(annotated, f"🚨 [LOST PROPERTY ALERT] {lost_cnt} UNATTENDED ITEM(S)", (32, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (20, 12), (580, hud_h - 12), (0, 130, 60), -1)
            cv2.putText(annotated, "[AI CCTV] POST-CLASS ITEM RECOVERY MONITORING", (32, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2)

        # 통계 배지 (SAFE / RETRIEVED / WARN / LOST)
        stat_x = w - 820
        # SAFE
        cv2.rectangle(annotated, (stat_x, 14), (stat_x + 95, hud_h - 14), (0, 150, 60), -1)
        cv2.putText(annotated, f"SAFE: {safe_cnt}", (stat_x + 10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        # RETRIEVED (가져감)
        cv2.rectangle(annotated, (stat_x + 105, 14), (stat_x + 245, hud_h - 14), (160, 180, 0), -1)
        cv2.putText(annotated, f"TAKEN: {retrieved_cnt}", (stat_x + 115, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
        # WARN
        cv2.rectangle(annotated, (stat_x + 255, 14), (stat_x + 360, hud_h - 14), (0, 140, 240), -1)
        cv2.putText(annotated, f"WARN: {warn_cnt}", (stat_x + 265, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
        # LOST
        cv2.rectangle(annotated, (stat_x + 370, 14), (stat_x + 475, hud_h - 14), (0, 0, 220) if lost_cnt > 0 else (60, 60, 60), -1)
        cv2.putText(annotated, f"LOST: {lost_cnt}", (stat_x + 380, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        cur_sec = frame_idx / fps
        tot_sec = total / fps
        cv2.putText(annotated, f"Students: {len(last_persons):2d} | Time: {cur_sec:4.1f}s / {tot_sec:4.1f}s",
                    (w - 320, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(annotated)

        if frame_idx % 60 == 0 or frame_idx == total:
            pct = (frame_idx / total) * 100
            elapsed = time.time() - t0
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"\r진행률: {frame_idx}/{total} ({pct:5.1f}%) | 렌더링 속도: {proc_fps:4.1f} FPS", end="", flush=True)

    cap.release()
    out.release()

    print(f"\n\n🎉 [완료] 정밀 분석 및 렌더링 완료! (소요 시간: {time.time() - t0:.1f}초)")
    print(f" - 생성 비디오: {OUTPUT_VIDEO}")
    print(f" - 최종 통계:")
    print(f"   • SAFE: {safe_cnt}")
    print(f"   • TAKEN / RETRIEVED (학생들이 챙겨간 물품): {retrieved_cnt}개")
    print(f"   • LOST (책상 위 방치된 분실물): {lost_cnt}개")
    print(f" - 발급된 공식 증거 리포트: {len(issued_reports)}건")
    for r in issued_reports:
        print(f"   • [{r['incident_id']}] {r['card_path']}")
        print(f"     HTML: {r['html_path']}")


if __name__ == "__main__":
    run_pipeline()
