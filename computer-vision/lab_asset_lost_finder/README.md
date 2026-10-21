# 🏫 실습실 기자재 반출 방지 및 분실물 탐지 시스템 (Lab Asset & Lost Finder)

> **CCTV 기반 영상 인식, FSM 상태 머신, SAHI 소형 객체 탐지 및 실시간 알림 파이프라인**

---

## 📌 주요 특징
1. **FSM (유한 상태 머신) 기반 이벤트 라이프사이클 관리**
   - **기자재 반출 방지**: `STABLE` (안전) ➡️ `IN_USE` (사용 중) ➡️ `MISSING` (분실 의심) ➡️ `SUSPECTED_THEFT` (도난 경보)
   - **개인 분실물 탐지**: `ACCOMPANIED` (소지자 동반) ➡️ `UNATTENDED` (방치 감지) ➡️ `LOST_CONFIRMED` (분실물 확정)
2. **SAHI (Slicing Aided Hyper Inference)**
   - 고해상도 CCTV 화면 내 소형 마우스, USB, HDMI 케이블 슬라이싱 추론 지원
3. **적응형 영상 전처리 (CLAHE)**
   - 실습실 조명 변화 및 역광 환경 대응
4. **FastAPI 백엔드 연동**
   - 이벤트 발생 시 실시간 대시보드 및 경보 전송

---

## 📁 주요 구성 파일
- `main.py`: 실시간 프레임 추론, FSM 룰 평가 및 HUD 시각화 실행
- `config.py`: ROI 관심영역(교탁, 출구문, 책상 등) 비율 및 타이머 설정
- `core/vision_engine.py`: YOLOv8/11 추론, CLAHE 전처리, 바운딩 박스 보정
- `core/state_machine.py`: 기자재 및 분실물 상태 머신 클래스
- `sahi_yolo_video_detection.py`: SAHI 기반 소형 객체 고정밀 비디오 분석
- `prepare_training_data.py`: 실습실 맞춤형 파인튜닝 데이터셋 생성 스크립트

---

## 🚀 실행 방법
```bash
# 기본 실시간 모니터링 실행
python main.py

# SAHI 비디오 탐지 실행
python sahi_yolo_video_detection.py
```
