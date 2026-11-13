# 설치와 실행

[메인 README](../../README.md)

## 준비물

- Python 3.10 이상 권장
- 웹캠 또는 분석용 영상
- YOLO 가중치 파일과 프로젝트 설정
- 모델 학습 시 데이터셋과 학습용 가중치

## 설치

~~~bash
git clone https://github.com/hyeonje3402/Object-Recognition-Vision.git
cd Object-Recognition-Vision/computer-vision/lab_asset_lost_finder
python -m venv .venv
~~~

Windows PowerShell에서는 가상환경을 활성화하고, 패키지를 설치합니다.

~~~powershell
.\.venv\Scripts\Activate.ps1
pip install ultralytics opencv-python sahi fastapi uvicorn numpy pillow
~~~

현재 프로젝트 폴더에는 requirements.txt가 확인되지 않아 위 명령은 기존 README에 기재된 핵심 패키지 설치 예시입니다. 실행 환경과 모델 구성에 따라 추가 패키지가 필요할 수 있습니다.

## 실행 예시

- 웹캠/실시간 프레임 처리: **python main.py**
- 상태 머신 시뮬레이션: **python demo_simulation.py**
- 실험용 통합 녹화 영상 파이프라인: **python run_integrated_lost_and_theft_pipeline.py**
- FastAPI 예제: **uvicorn api.server:app --reload --port 8000**

통합 영상 스크립트는 특정 입력 영상 파일 경로를 참조합니다. 해당 영상과 가중치 파일이 로컬에 준비됐는지 먼저 확인하세요. API 서버 문서는 실행 후 **http://127.0.0.1:8000/docs**에서 볼 수 있습니다.

## 설정

config.py의 ROI 좌표는 프레임 크기에 대한 비율로 설정되며, 카메라 구도에 맞춰 교탁·출입구·책상 구역을 조정해야 합니다. 타이머와 사람-물품 근접 거리도 실험 목적에 따라 설정합니다. README의 짧은 타이머 값은 데모용이며 실제 관제에 그대로 적용하면 안 됩니다.

## 결과물과 모델 파일

분석 결과는 프로젝트 내 output_results/ 하위에 저장됩니다. 학습 스크립트는 Ultralytics YOLO11n을 사용해 커스텀 데이터셋을 학습하고 runs/ 아래에 가중치와 학습 산출물을 남깁니다. 대형 모델·영상 파일은 저장소에 모두 포함되어 있지 않을 수 있으므로 입력 파일 존재 여부를 확인하세요.