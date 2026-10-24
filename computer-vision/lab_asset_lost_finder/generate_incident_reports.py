"""
[실제 사건 시뮬레이션 및 공식 증거 리포트 생성 스크립트 - generate_incident_reports.py]
2가지 핵심 시나리오(분실물 방치 & 공용 기자재 도난)에 대해:
1. 증거 물품 고화질 크롭 이미지 생성
2. 비주얼 인스펙션 리포트 카드(JPG) 렌더링
3. 브라우저 인쇄용 공식 HTML 리포트 발급
4. 데이터 연동용 JSON 메타데이터 생성
"""
import sys
import cv2
from pathlib import Path
from core.report_generator import IncidentReportGenerator

BASE_DIR = Path(__file__).resolve().parent
FRAME_PATH = BASE_DIR / "scratch_preview" / "new_video_frame_0_0.jpg"
if not FRAME_PATH.exists():
    FRAME_PATH = BASE_DIR / "input_images" / "lab_classroom_cctv.jpg"

def run():
    print("=" * 65)
    print("[SMART LAB] Incident Report & Evidence Generation Start")
    print("=" * 65)

    img = cv2.imread(str(FRAME_PATH))
    if img is None:
        print("[ERROR] CCTV frame not found at:", FRAME_PATH)
        return

    reporter = IncidentReportGenerator()

    # =========================================================================
    # 사건 1: 학생 퇴실 후 3열 2번 좌석 에어팟(AirPods) 방치 분실물 확정
    # =========================================================================
    print("\n[Case 1] Generating Lost Property Report (AirPods at Seat #3)...")
    bbox_airpods = [1985, 785, 2045, 835]
    
    rep1 = reporter.generate_incident_report(
        full_frame=img,
        bbox=bbox_airpods,
        incident_type="LOST_PROPERTY",
        item_class="airpods",
        confidence=0.924,
        classroom_id="IT-융합공학관 402호 멀티미디어실습실",
        camera_id="CCTV-CAM-01 (후방 4K 와이드)",
        location_desc="3열 2번 좌석 (키보드 우측 마우스패드 옆)",
        suspect_desc="23:35분경 네이비 후드티 착용 학생 퇴실 후 책상 위에 방치됨. 유예 시간(5분) 초과",
        action_note="실습실 관리자 분실물 임시 보관함 #A-03으로 이관 및 학생 포털 분실물 게시판 자동 등록"
    )

    # =========================================================================
    # 사건 2: 교탁 공용 마우스(Public Mouse) 무단 반출 도난 경보
    # =========================================================================
    print("\n[Case 2] Generating Asset Theft Report (Podium Public Mouse)...")
    bbox_mouse = [1120, 990, 1175, 1045]

    rep2 = reporter.generate_incident_report(
        full_frame=img,
        bbox=bbox_mouse,
        incident_type="ASSET_THEFT",
        item_class="public_mouse",
        confidence=0.968,
        classroom_id="IT-융합공학관 402호 멀티미디어실습실",
        camera_id="CCTV-CAM-01 (후방 4K 와이드)",
        location_desc="강단 교탁 메인 PC 본체 우측 거치대",
        suspect_desc="수업 종료 직후 교탁에 교수가 없는 상태에서 물품이 분리되어 출입문(DOOR ROI)으로 이동 감지",
        action_note="경비 관제실 비상 팝업 알림 전송 및 해당 타임스탬프 CCTV 녹화 영상(10초) 영구 보존 조치"
    )

    print("\n" + "=" * 65)
    print("[SUCCESS] 2 Incident Reports Successfully Generated!")
    print(f" 1. Lost Property Visual Card: {rep1['card_path'].name}")
    print(f" 2. Asset Theft Visual Card:   {rep2['card_path'].name}")
    print(f" 3. HTML Reports Folder:       {reporter.html_dir}")
    print("=" * 65)

if __name__ == "__main__":
    run()
