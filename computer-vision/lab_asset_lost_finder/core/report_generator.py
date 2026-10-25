"""
[실습실 사건 사고 증거 리포트 생성기 - core/report_generator.py]
분실물(LOST) 또는 공용 기자재 도난(THEFT) 발생 시:
1. 해당 물품 고화질 바운딩 박스 크롭(Evidence Zoom-in) 자동 추출
2. 강의실 정보(건물명, 호수, 카메라 ID), 날짜, 시각, 물품 정보 메타데이터 기록
3. 인쇄/제출 가능한 시각적 검증 카드 이미지(JPG - 맑은 고딕 한글 렌더링) 생성
4. 웹 브라우저 인쇄용 공식 사고 보고서(HTML) 및 구조화 데이터(JSON) 자동 발급
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class IncidentReportGenerator:
    def __init__(self, output_base_dir: Optional[Path] = None):
        if output_base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent / "output_results" / "incident_reports"
        else:
            self.base_dir = Path(output_base_dir)

        self.crops_dir = self.base_dir / "crops"
        self.cards_dir = self.base_dir / "cards"
        self.html_dir = self.base_dir / "html"
        self.json_dir = self.base_dir / "json"

        for d in [self.crops_dir, self.cards_dir, self.html_dir, self.json_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.font_path = "C:/Windows/Fonts/malgun.ttf"
        if not os.path.exists(self.font_path):
            self.font_path = "C:/Windows/Fonts/gulim.ttc"

    def _draw_text_kr(self, img_bgr: np.ndarray, text: str, xy: Tuple[int, int], size: int = 18, color_bgr: Tuple[int, int, int] = (255, 255, 255), bold: bool = False) -> np.ndarray:
        """PIL을 사용하여 OpenCV BGR 이미지 위에 선명한 한글 텍스트 렌더링"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_im)
        try:
            font = ImageFont.truetype(self.font_path, size)
        except Exception:
            font = ImageFont.load_default()
        
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(xy, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    def generate_incident_report(
        self,
        full_frame: np.ndarray,
        bbox: List[int],               # [x1, y1, x2, y2]
        incident_type: str,            # "LOST_PROPERTY" 또는 "ASSET_THEFT"
        item_class: str,               # "airpods", "charger", "public_mouse", "laptop" 등
        confidence: float,             # 0.0 ~ 1.0
        classroom_id: str = "IT-융합공학관 402호 멀티미디어실습실",
        camera_id: str = "CCTV-CAM-01 (후방 와이드)",
        location_desc: str = "3열 2번 좌석 책상 위",
        suspect_desc: Optional[str] = "출입문 방향으로 이동한 회색 후드티 사용자",
        action_note: str = "실습실 분실물 보관함 #03 이관 완료 및 학과 사무실 등록",
        crop_source_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        단일 사건에 대한 증거 크롭, 비주얼 리포트 카드, HTML 문서, JSON 파일 원클릭 일괄 생성
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")
        incident_id = f"INC-{file_timestamp}-{abs(hash(item_class + location_desc)) % 1000:03d}"

        h, w, _ = full_frame.shape
        x1, y1, x2, y2 = bbox

        # 1. 고화질 증거 물품 크롭 (작은 물품은 넉넉한 마진 적용)
        margin = 55 if item_class.lower() in ["mouse", "airpods", "charger"] else 35
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(w, x2 + margin)
        crop_y2 = min(h, y2 + margin)
        # crop_source_frame이 주어지면(물품이 아직 존재하던 기준 프레임) 그 프레임에서 증거 크롭 추출
        crop_src = crop_source_frame if crop_source_frame is not None else full_frame
        crop_img = crop_src[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # 크롭 이미지 저장
        crop_filename = f"{incident_id}_{item_class}_crop.jpg"
        crop_path = self.crops_dir / crop_filename
        cv2.imwrite(str(crop_path), crop_img)

        # 2. 비주얼 증거 리포트 카드 이미지 (JPG) 생성
        card_img = self._create_visual_report_card(
            full_frame=full_frame,
            crop_img=crop_img,
            bbox=bbox,
            incident_id=incident_id,
            incident_type=incident_type,
            item_class=item_class,
            confidence=confidence,
            classroom_id=classroom_id,
            camera_id=camera_id,
            location_desc=location_desc,
            timestamp_str=timestamp_str,
            suspect_desc=suspect_desc,
            action_note=action_note
        )
        card_filename = f"{incident_id}_{incident_type}_report_card.jpg"
        card_path = self.cards_dir / card_filename
        cv2.imwrite(str(card_path), card_img)

        # 3. JSON 구조화 메타데이터 저장
        metadata = {
            "incident_id": incident_id,
            "incident_type": incident_type,
            "item_class": item_class,
            "confidence": round(confidence, 4),
            "timestamp": timestamp_str,
            "classroom": {
                "name": classroom_id,
                "camera": camera_id,
                "seat_location": location_desc
            },
            "bounding_box": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1
            },
            "investigation": {
                "suspect_note": suspect_desc,
                "action_taken": action_note
            },
            "files": {
                "crop_image": str(crop_path.relative_to(self.base_dir.parent)),
                "report_card_image": str(card_path.relative_to(self.base_dir.parent))
            }
        }
        json_path = self.json_dir / f"{incident_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 4. 브라우저 인쇄용 공식 HTML 리포트 문서 생성
        html_content = self._create_html_report(metadata, crop_filename, card_filename)
        html_path = self.html_dir / f"{incident_id}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return {
            "incident_id": incident_id,
            "crop_path": crop_path,
            "card_path": card_path,
            "html_path": html_path,
            "json_path": json_path,
            "metadata": metadata
        }

    def _create_visual_report_card(
        self,
        full_frame: np.ndarray,
        crop_img: np.ndarray,
        bbox: List[int],
        incident_id: str,
        incident_type: str,
        item_class: str,
        confidence: float,
        classroom_id: str,
        camera_id: str,
        location_desc: str,
        timestamp_str: str,
        suspect_desc: Optional[str],
        action_note: str
    ) -> np.ndarray:
        """
        현장 전체 사진(발생 좌표 하이라이트) + 확대 크롭 샷 + 공식 양식 텍스트가 결합된 1400x920 비주얼 카드
        """
        card_w, card_h = 1400, 920
        card = np.full((card_h, card_w, 3), 24, dtype=np.uint8)

        is_theft = incident_type in ["ASSET_THEFT", "PERSONAL_ITEM_THEFT", "PERSONAL_THEFT"]
        theme_color = (0, 0, 255) if is_theft else (0, 165, 255)  # 도난: 빨강, 분실: 주황
        if incident_type == "PERSONAL_ITEM_THEFT":
            status_title = "[도난 경보 THEFT ALERT] 개인 소지품(무선이어폰/AirPods) 도난 조사서"
        elif is_theft:
            status_title = "[도난 경보 THEFT ALERT] 공용 기자재 무단 반출 조사서"
        else:
            status_title = "[분실물 방치 LOST PROPERTY] 개인 소지품 습득 조사서"

        # 1. 상단 공식 헤더 배너
        cv2.rectangle(card, (0, 0), (card_w, 90), (32, 32, 38), -1)
        cv2.rectangle(card, (0, 88), (card_w, 90), theme_color, -1)
        
        card = self._draw_text_kr(card, status_title, (30, 18), size=26, color_bgr=(255, 255, 255))
        card = self._draw_text_kr(card, f"SMART LAB CCTV AI INCIDENT VERIFICATION SYSTEM | 사건 번호: {incident_id}",
                                  (30, 58), size=14, color_bgr=(180, 180, 180))

        # 2. 좌측: 전체 CCTV 현장 프레임 (타겟 하이라이트)
        scene_w, scene_h = 780, 480
        scene_view = full_frame.copy()
        x1, y1, x2, y2 = bbox
        # 발생 위치 하이라이트 사각 박스 & 펄스 효과
        cv2.rectangle(scene_view, (x1, y1), (x2, y2), theme_color, 3)
        cv2.circle(scene_view, ((x1 + x2) // 2, (y1 + y2) // 2), 18, theme_color, 2)
        cv2.putText(scene_view, f"TARGET: {item_class.upper()}", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme_color, 2)

        scene_resized = cv2.resize(scene_view, (scene_w, scene_h))
        card[120:120 + scene_h, 30:30 + scene_w] = scene_resized
        cv2.rectangle(card, (30, 120), (30 + scene_w, 120 + scene_h), (80, 80, 80), 1)
        card = self._draw_text_kr(card, "[CCTV 현장 전체 감시 프레임 - 발생 위치 하이라이트]", (30, 96), size=14, color_bgr=(200, 200, 200))

        # 3. 우측 상단: 확대 크롭 샷 (Evidence Zoom-in)
        crop_disp_w, crop_disp_h = 360, 360
        crop_resized = cv2.resize(crop_img, (crop_disp_w, crop_disp_h))
        crop_x = 850
        crop_y = 150
        card[crop_y:crop_y + crop_disp_h, crop_x:crop_x + crop_disp_w] = crop_resized
        cv2.rectangle(card, (crop_x, crop_y), (crop_x + crop_disp_w, crop_y + crop_disp_h), theme_color, 2)
        card = self._draw_text_kr(card, "[증거 물품 고화질 확대 크롭 샷 (Evidence Zoom)]", (crop_x, crop_y - 28), size=15, color_bgr=theme_color)

        # 크롭 아래 정밀 태그
        card = self._draw_text_kr(card, f"탐지 물품: {item_class.upper()} | 신뢰도: {confidence * 100:.1f}%",
                                  (crop_x, crop_y + crop_disp_h + 12), size=16, color_bgr=(255, 255, 255))

        # 4. 하단 공식 조사 메타데이터 정보 테이블
        table_y = 620
        cv2.rectangle(card, (30, table_y), (card_w - 30, card_h - 20), (32, 34, 40), -1)
        cv2.rectangle(card, (30, table_y), (card_w - 30, card_h - 20), (70, 75, 85), 1)

        # 2열 정보 (상단 3행)
        top_rows = [
            ("발생 일시", timestamp_str, "설치 강의실", classroom_id),
            ("사건 유형", incident_type, "관제 카메라", camera_id),
            ("대상 물품", f"{item_class.upper()} (신뢰도: {confidence * 100:.1f}%)", "상세 위치", location_desc),
        ]
        for i, (l1, v1, l2, v2) in enumerate(top_rows):
            curr_y = table_y + 16 + i * 42
            card = self._draw_text_kr(card, f"• {l1}:", (45, curr_y), size=14, color_bgr=(160, 160, 160))
            card = self._draw_text_kr(card, str(v1), (150, curr_y), size=15, color_bgr=(255, 255, 255))
            card = self._draw_text_kr(card, f"• {l2}:", (680, curr_y), size=14, color_bgr=(160, 160, 160))
            card = self._draw_text_kr(card, str(v2), (790, curr_y), size=15, color_bgr=(255, 255, 255))

        # 하단 전체 너비 행 (조사 소견 & 후속 조치)
        row4_y = table_y + 16 + 3 * 42
        card = self._draw_text_kr(card, "• 조사 소견:", (45, row4_y), size=14, color_bgr=(160, 160, 160))
        card = self._draw_text_kr(card, str(suspect_desc if suspect_desc else "특이사항 없음"), (150, row4_y), size=14, color_bgr=(230, 230, 230))

        row5_y = table_y + 16 + 4 * 42
        card = self._draw_text_kr(card, "• 후속 조치:", (45, row5_y), size=14, color_bgr=(160, 160, 160))
        card = self._draw_text_kr(card, str(action_note), (150, row5_y), size=15, color_bgr=(0, 255, 200))

        return card

    def _create_html_report(self, meta: Dict[str, Any], crop_fn: str, card_fn: str) -> str:
        """브라우저에서 바로 열고 A4 출력/PDF 저장 가능한 깔끔한 웹 리포트"""
        is_theft = meta["incident_type"] in ["ASSET_THEFT", "PERSONAL_ITEM_THEFT", "PERSONAL_THEFT"]
        badge_color = "#e53935" if is_theft else "#fb8c00"
        if meta["incident_type"] == "PERSONAL_ITEM_THEFT":
            title = "개인 소지품 도난 조사 보고서 (무선이어폰/AirPods)"
        elif is_theft:
            title = "공용 기자재 도난 조사 보고서"
        else:
            title = "실습실 분실물 습득 조사 보고서"

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>[공식 리포트] {meta['incident_id']} - {title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Malgun Gothic", "Segoe UI", Roboto, sans-serif; background: #f0f2f5; margin: 0; padding: 30px; }}
  .container {{ max-width: 900px; margin: 0 auto; background: #fff; border-radius: 10px; box-shadow: 0 4px 18px rgba(0,0,0,0.08); overflow: hidden; }}
  .header {{ background: #1a1e24; color: #fff; padding: 25px 35px; border-bottom: 4px solid {badge_color}; }}
  .header h1 {{ margin: 0; font-size: 24px; display: flex; align-items: center; gap: 12px; }}
  .badge {{ background: {badge_color}; color: #fff; font-size: 13px; padding: 4px 10px; border-radius: 4px; font-weight: bold; }}
  .meta-sub {{ margin-top: 8px; color: #a0aec0; font-size: 13px; }}
  .content {{ padding: 30px 35px; }}
  .section-title {{ font-size: 17px; font-weight: bold; color: #2d3748; margin-bottom: 15px; border-left: 4px solid {badge_color}; padding-left: 10px; }}
  .table-info {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
  .table-info th, .table-info td {{ padding: 12px 14px; border: 1px solid #e2e8f0; font-size: 14px; }}
  .table-info th {{ background: #f7fafc; color: #4a5568; width: 22%; text-align: left; }}
  .evidence-grid {{ display: flex; gap: 20px; margin-bottom: 30px; }}
  .evidence-box {{ flex: 1; background: #fafafa; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; text-align: center; }}
  .evidence-box img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #cbd5e0; }}
  .evidence-title {{ font-size: 13px; font-weight: bold; color: #4a5568; margin-bottom: 8px; }}
  .footer {{ background: #f8fafc; padding: 18px 35px; border-top: 1px solid #edf2f7; font-size: 12px; color: #718096; display: flex; justify-content: space-between; align-items: center; }}
  .print-btn {{ background: #2b6cb0; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; }}
  @media print {{ body {{ background: #fff; padding: 0; }} .container {{ box-shadow: none; border-radius: 0; }} .print-btn {{ display: none; }} }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{title} <span class="badge">{meta['incident_type']}</span></h1>
    <div class="meta-sub">사건 고유 관리 번호: <strong>{meta['incident_id']}</strong> | CCTV AI 자동 관제 엔진 생성</div>
  </div>
  
  <div class="content">
    <div class="section-title">1. 사건 및 발생 장소 개요</div>
    <table class="table-info">
      <tr>
        <th>설치 강의실</th>
        <td><strong>{meta['classroom']['name']}</strong></td>
        <th>관제 카메라</th>
        <td>{meta['classroom']['camera']}</td>
      </tr>
      <tr>
        <th>발생 일시</th>
        <td>{meta['timestamp']}</td>
        <th>상세 위치</th>
        <td><strong>{meta['classroom']['seat_location']}</strong></td>
      </tr>
      <tr>
        <th>탐지 물품</th>
        <td><strong style="color: {badge_color};">{meta['item_class'].upper()}</strong> (인공지능 신뢰도: {meta['confidence']*100:.1f}%)</td>
        <th>바운딩 좌표</th>
        <td>X: {meta['bounding_box']['x1']}~{meta['bounding_box']['x2']}, Y: {meta['bounding_box']['y1']}~{meta['bounding_box']['y2']}</td>
      </tr>
    </table>

    <div class="section-title">2. CCTV 영상 기반 시각적 증거물 (Evidence Media)</div>
    <div class="evidence-grid">
      <div class="evidence-box" style="flex: 2;">
        <div class="evidence-title">전체 현장 검증 스냅샷 (발생 위치 하이라이트)</div>
        <img src="../cards/{card_fn}" alt="Full Incident Card">
      </div>
      <div class="evidence-box" style="flex: 1;">
        <div class="evidence-title">증거 물품 확대 크롭 (Zoom-In)</div>
        <img src="../crops/{crop_fn}" alt="Zoom-in Crop">
      </div>
    </div>

    <div class="section-title">3. 조사 소견 및 행정 조치 내역</div>
    <table class="table-info">
      <tr>
        <th>조사 소견</th>
        <td>{meta['investigation']['suspect_note']}</td>
      </tr>
      <tr>
        <th>후속 행정 조치</th>
        <td><strong style="color: #2b6cb0;">{meta['investigation']['action_taken']}</strong></td>
      </tr>
    </table>
  </div>

  <div class="footer">
    <div>스마트 실습실 안전·보안 관제 센터 | 자동 발급 시스템</div>
    <button class="print-btn" onclick="window.print()">🖨️ 보고서 인쇄 / PDF 저장</button>
  </div>
</div>
</body>
</html>
"""
        return html
