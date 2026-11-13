# 아키텍처와 판정 흐름

[메인 README](../../README.md)

## 데이터 흐름

1. 웹캠 또는 녹화 영상을 OpenCV로 프레임 단위 입력합니다.
2. 비전 엔진이 CLAHE 전처리 후 YOLO 모델로 사람과 물품을 검출합니다. 소형·원거리 물체를 위한 추가 크롭 및 SAHI 기반 스크립트도 포함되어 있습니다.
3. 사람과 물품의 위치를 교탁·출입구·학생 책상 ROI 및 추적 정보와 비교합니다.
4. 상태 머신이 잠깐의 가림, 물품의 부재 시간, 사람의 위치를 고려해 정상 사용·주의·반출 경보 또는 방치 물품 상태를 계산합니다.
5. HUD와 사건 보고서 생성기가 결과를 이미지·HTML·JSON으로 기록합니다.

## 주요 모듈

| 경로 | 역할 |
|---|---|
| computer-vision/lab_asset_lost_finder/main.py | 프레임 처리와 HUD가 포함된 실시간 진입점 |
| core/vision_engine.py | YOLO 추론, CLAHE 전처리, 소형 객체 탐지 보조 |
| core/state_machine.py | 공용 자산과 방치 물품 상태 전이 |
| core/interaction_tracker.py | 사람·물품 간 상호작용 및 이동 기반 판정 |
| core/personal_item_tracker.py | 개인 소지품 소유 관계와 상호작용 추적 |
| core/two_stage_item_classifier.py | 작은 물체 후보의 2차 분류 |
| core/report_generator.py | 사건 카드와 HTML/JSON 보고서 생성 |
| api/server.py | 상태 REST API와 WebSocket 경보 연결 예제 |

## 판정 개념

- 공용 기자재는 교탁 ROI에서 보이지 않더라도 사용자가 교탁에 있을 때 즉시 도난으로 판단하지 않도록 FSM에 사용 상태를 둡니다.
- 개인 소지품은 사람과 물품의 근접 상태, 자리 이탈, 타이머를 사용해 방치 여부를 추적합니다.
- ROI 좌표와 데모 타이머는 config.py에서 조정합니다. 현재 설정값은 실제 공간의 보안 기준이나 검증된 운영 임계값으로 간주하면 안 됩니다.

## 저장과 연동 경계

분석 결과, 사건 카드, 영상 캐시는 로컬 파일로 저장합니다. 저장소에서 PostgreSQL, 클라우드 객체 스토리지, 사용자 인증 서비스는 확인되지 않습니다. FastAPI 예제는 상태 조회와 WebSocket 연결을 제공하지만 로컬 비전 파이프라인의 모든 사건이 서버로 자동 전달되는 운영 통합은 별도 작업입니다.