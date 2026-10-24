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
- `run_real_theft_incident_pipeline.py`: 실전 CCTV 비디오 기반 동적 상호작용 도난 감지 및 증거 리포트 자동 발급 파이프라인
- `core/interaction_tracker.py`: ByteTrack + 템플릿 매칭 결합 동적 도난 판정 엔진 (프레임 하드코딩 제거)
- `core/personal_item_tracker.py`: 개인 소지품 소유자 귀속 및 제3자 절도 판정 엔진 (엣지케이스 방어)
- `core/report_generator.py`: 분실물/도난 증거 물품 고화질 크롭 및 공식 조사 보고서(JPG/HTML/JSON) 자동 생성기
- `tests/test_personal_item_theft.py`: 개인 소지품 절도/방치 시나리오 10종 엣지케이스 단위 테스트
- `config.py`: ROI 관심영역(교탁, 출구문, 책상 등) 비율 및 타이머 설정
- `generate_incident_reports.py`: 다중 강의실 분실물 및 도난 사건 증거 리포트 자동 발급 실증 스크립트
- `render_all_items_new_video.py`: 2944x1248 초고화질 신규 CCTV 영상 전수 객체 탐지 및 창문 반사 필터링
- `prepare_training_data.py`: 실습실 맞춤형 초소형 분실물(에어팟, 충전기/어댑터) 데이터셋 증강 및 패치 생성
- `train_lab_model.py`: YOLO11n 기반 실습실 전이학습(Fine-tuning) 스크립트
- `evaluate_trained_model.py`: 파인튜닝된 가중치(`best.pt`) 추론 및 HUD 통계 시각화

---

## 📑 주간 실습 및 중간평가 보고서 (`docs/reports/`)
- [3주차 주간 보고서 (수치화)](docs/reports/week3_report.md): FSM 3단계 상태 머신, ROI 크롭 & CLAHE 전처리
- [4주차 주간 보고서 (수치화)](docs/reports/week4_report.md): 전 좌석 전수 인벤토리 HUD 모니터링 및 원거리 미탐 분석
- [5주차 주간 보고서 (수치화)](docs/reports/week5_report.md): SAHI 타일 슬라이싱, Grid Prior 융합, 650장 파인튜닝, ByteTrack 버퍼
- [6주차 주간 보고서 (수치화)](docs/reports/week6_report.md): 2944×1248 듀얼 실영상 CCTV 파이프라인, RGB 색상 보정, GREEDYNMM(IOS)
- [7주차 중간평가 보고서 (수치화)](docs/reports/week7_midterm_report.md): 동적 FSM 도난 판정, 3종 증거 리포트 자동 발급, 소유권 추적 단위 테스트

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
