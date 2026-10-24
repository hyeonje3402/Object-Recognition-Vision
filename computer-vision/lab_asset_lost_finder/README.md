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
- `prepare_training_data.py`: 실습실 맞춤형 초소형 분실물(에어팟, 충전기/어댑터) 데이터셋 증강 및 패치 생성
- `train_lab_model.py`: YOLO11n 기반 실습실 전이학습(Fine-tuning) 스크립트
- `evaluate_trained_model.py`: 파인튜닝된 가중치(`best.pt`) 추론 및 HUD 통계 시각화

---

## 🎯 커스텀 파인튜닝 모델 (Option 4 완료)
COCO 기본 모델이 인식하기 어려운 실습실 내 초소형 분실물 전용 클래스 확장 및 학습:
- `mouse`: 학생 좌석 마우스
- `public_mouse`: 교탁 고정 배치 공용 마우스 (도난 감시대상)
- `keyboard`: 키보드
- `airpods`: 무선 이어폰/에어팟 충전 케이스 (초소형 빈출 분실물)
- `charger`: 스마트폰/노트북 충전 어댑터 및 C타입 케이블

---

## 🚀 실행 방법
```bash
# 1. 초소형 분실물 특화 데이터셋 증강 생성
python prepare_training_data.py

# 2. YOLO11n 모델 커스텀 파인튜닝
python train_lab_model.py

# 3. 학습 모델 평가 및 시각화 HUD 렌더링
python evaluate_trained_model.py

# 4. 실시간 통합 관제 시스템 실행
python main.py
```
