# 🏫 실습실 기자재 반출 방지 및 분실물 탐지 시스템 (Lab Asset & Lost Finder)

> **CCTV 기반 영상 인식, 2단계 크롭 정밀 분류기(AirPods/무선이어폰), FSM 상태 머신, SAHI 소형 객체 탐지 및 실시간 도난/분실 알림 파이프라인**

---

## 📌 주요 특징
1. **2단계 크롭 정밀 분류기 (Two-Stage Crop Classifier)**
   - YOLO 기본 COCO 80종 모델의 한계(무선이어폰/에어팟 클래스 부재로 인한 마우스/리모컨 오분류)를 완벽 극복
   - 크롭 패치의 **기하학적 형상(Aspect Ratio 1.12 스퀘어클) + 표면 질감/색상(초저채도 Sat 20.8, 고휘도 Val 236.3) + MobileNetV3 딥러닝 임베딩(화장품/소형 케이스 40.8%, 마우스 0.0%)**을 융합하여 **`무선이어폰 (AirPods)`을 93.4% 신뢰도로 정확히 식별**
2. **동적 상호작용 도난 판정 엔진 (`DynamicTheftDetector`)**
   - 24.0초 교탁 위 무선이어폰(에어팟) 파지 및 출입문(DOOR ROI) 방향 급격한 이동(543.8px 이탈) 실시간 포착
   - 용의자(ByteTrack ID #102) 실시간 추적 박스 적색 전환 및 상단 경보 HUD 배너 즉시 발령
3. **학생 퇴실 및 방치 분실물 라이프사이클 관리 (FSM)**
   - 학생들이 본인 짐을 챙겨 나가는 정상 퇴실 행위는 `TAKEN / RETRIEVED`로 자동 소멸시켜 **빈 책상 허위 박스(Ghost Box) 완전 제거** (68건 정상 회수 처리)
   - 수업 종료 후 책상 위에 남겨진 음료/텀블러 2건만 정확하게 `[LOST PROPERTY]`로 식별 및 공식 분실물 접수
4. **공식 증거 조사 보고서 자동 발급 시스템 (`IncidentReportGenerator`)**
   - 사건 발생 즉시 **현장 전체 스냅샷 + 고화질 증거 크롭 샷(Evidence Zoom-in) + 공식 서식 메타데이터**가 결합된 1400×920 비주얼 리포트 카드(JPG), 브라우저 A4 인쇄용 공식 웹 리포트(HTML), 구조화 데이터(JSON) 원클릭 동시 발행

---

## 📁 주요 구성 파일
- **`run_integrated_lost_and_theft_pipeline.py`**: **[핵심 실전 파이프라인]** 24초 교탁 에어팟 도난 검거 + 퇴실 방치 분실물 감지 + 공식 보고서 자동 발급 통합 마스터 스크립트
- **`core/two_stage_item_classifier.py`**: **[신규]** 미세 소형 객체(무선이어폰/에어팟 vs 마우스 vs 리모컨) 2단계 정밀 분류 모듈
- **`core/interaction_tracker.py`**: ByteTrack + 템플릿 매칭 결합 동적 도난 판정 엔진 (프레임 하드코딩 제거)
- **`core/personal_item_tracker.py`**: 개인 소지품 소유자 귀속 및 제3자 절도 판정 엔진 (소유권 이전/대리 챙김 방어)
- **`core/report_generator.py`**: 분실물(`LOST_PROPERTY`) 및 개인 소지품 도난(`PERSONAL_ITEM_THEFT`), 공용 자산 도난(`ASSET_THEFT`) 3종 공식 조사 보고서(JPG/HTML/JSON) 자동 생성기
- **`tests/test_personal_item_theft.py`**: 개인 소지품 절도/방치 시나리오 10종 엣지케이스 단위 테스트
- **`main.py`**: 실시간 프레임 추론, FSM 룰 평가 및 HUD 시각화 실행
- **`config.py`**: ROI 관심영역(교탁, 출구문, 책상 등) 비율 및 타이머 설정
- **`prepare_training_data.py`**: 실습실 맞춤형 초소형 분실물(에어팟, 충전기/어댑터) 데이터셋 증강 및 패치 생성
- **`train_lab_model.py`**: YOLO11n 기반 실습실 전이학습(Fine-tuning) 스크립트
- **`evaluate_trained_model.py`**: 파인튜닝된 가중치(`best.pt`) 추론 및 HUD 통계 시각화

---

## 📑 주간 실습 및 중간평가 보고서 (`docs/reports/`)
- [3주차 주간 보고서 (수치화)](docs/reports/week3_report.md): FSM 3단계 상태 머신, ROI 크롭 & CLAHE 전처리
- [4주차 주간 보고서 (수치화)](docs/reports/week4_report.md): 전 좌석 전수 인벤토리 HUD 모니터링 및 원거리 미탐 분석
- [5주차 주간 보고서 (수치화)](docs/reports/week5_report.md): SAHI 타일 슬라이싱, Grid Prior 융합, 650장 파인튜닝, ByteTrack 버퍼
- [6주차 주간 보고서 (수치화)](docs/reports/week6_report.md): 2944×1248 듀얼 실영상 CCTV 파이프라인, RGB 색상 보정, GREEDYNMM(IOS)
- [7주차 중간평가 보고서 (수치화)](docs/reports/week7_midterm_report.md): 동적 FSM 도난 판정, 3종 증거 리포트 자동 발급, 소유권 추적 단위 테스트

---

## 🎯 2단계 정밀 분류 및 실증 사건 보고서 예시

### 1. 24.0초 무선이어폰(AirPods) 도난 조사서 (PERSONAL_ITEM_THEFT)
* **사건 번호**: `INC-20261026_014228-612`
* **대상 물품**: `무선이어폰 (AirPods)` (신뢰도: **93.4%**)
* **용의자 식별**: 남색 맨투맨 및 회색 백팩 착용 학생 (`ByteTrack ID #102`)
* **발급 결과물**:
  - 카드 이미지: `output_results/incident_reports/cards/INC-20261026_014228-612_PERSONAL_ITEM_THEFT_report_card.jpg`
  - 공식 웹 리포트: `output_results/incident_reports/html/INC-20261026_014228-612.html`
  - 메타데이터 JSON: `output_results/incident_reports/json/INC-20261026_014228-612.json`

### 2. 2열 책상 방치 음료/텀블러 분실물 조사서 (LOST_PROPERTY)
* **사건 번호**: `INC-20261026_014145-143`
* **대상 물품**: `BOTTLE/TUMBLER #2` (신뢰도: **94.0%**)
* **발급 결과물**:
  - 카드 이미지: `output_results/incident_reports/cards/INC-20261026_014145-143_LOST_PROPERTY_report_card.jpg`
  - 공식 웹 리포트: `output_results/incident_reports/html/INC-20261026_014145-143.html`

---

## 🚀 실행 방법
```bash
# 1. 24초 에어팟 도난 및 방치 분실물 통합 실전 관제 파이프라인 실행
python run_integrated_lost_and_theft_pipeline.py

# 2. 개인 소지품 도난 엣지케이스 단위 테스트 실행
pytest tests/test_personal_item_theft.py -v

# 3. 실시간 웹캠/스트림 관제 메인 실행
python main.py
```
