"""
[실습실 특화 객체 탐지 모델 학습 스크립트 - train_lab_model.py]
YOLO11 Nano 모델을 기반으로 실습실 환경의 5대 핵심 물품:
- mouse (마우스)
- public_mouse (교탁 공용 마우스)
- keyboard (키보드)
- airpods (에어팟 / 무선 이어폰)
- charger (충전 어댑터 / 케이블)
데이터셋을 파인튜닝(전이학습)합니다.
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
PRETRAINED_MODEL = BASE_DIR / "yolo11n.pt"

def train():
    print("=" * 65)
    print("🚀 [AI Training] Custom Fine-Tuning YOLO11 for Small Lost Items")
    print(" - Target Classes: mouse, public_mouse, keyboard, airpods, charger")
    print(f" - Base Weights: {PRETRAINED_MODEL}")
    print(f" - Dataset Config: {DATA_YAML}")
    print("=" * 65)

    if not DATA_YAML.exists():
        print(f"[오류] 데이터셋 설정 파일이 없습니다: {DATA_YAML}")
        return

    # 1. 모델 로드 (Pretrained YOLO11n)
    model = YOLO(str(PRETRAINED_MODEL))

    # 2. 파인튜닝 학습 실행
    results = model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=640,
        batch=8,
        workers=0,
        device="cpu",
        warmup_epochs=1.0,
        lr0=0.01,
        project=str(BASE_DIR / "runs"),
        name="lab_custom_small_items",
        exist_ok=True,
        verbose=True,
        plots=True
    )

    best_weights = BASE_DIR / "runs" / "lab_custom_small_items" / "weights" / "best.pt"
    print("\n" + "=" * 65)
    print("🎉 [AI Training Complete] Finished fine-tuning lab model!")
    print(f" - Best weights saved at: {best_weights}")
    print("=" * 65)

if __name__ == "__main__":
    train()
