"""
[실습실 특화 객체 탐지 모델 학습 스크립트 - train_lab_model.py]
YOLO11 Nano 모델을 기반으로 실습실 환경의 마우스, 공용마우스, 키보드, 소형 분실물 데이터셋을 파인튜닝(전이학습)합니다.
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
PRETRAINED_MODEL = BASE_DIR / "yolo11n.pt"

def train():
    print("=" * 60)
    print("[AI Training] Custom Fine-Tuning YOLO for Lab Equipment & Mice")
    print(f" - Base Weights: {PRETRAINED_MODEL}")
    print(f" - Dataset Config: {DATA_YAML}")
    print("=" * 60)

    # 1. 모델 로드 (Pretrained YOLO11n)
    model = YOLO(str(PRETRAINED_MODEL))

    # 2. 파인튜닝 학습 실행
    results = model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=640,
        batch=4,
        workers=0,
        warmup_epochs=1.0,
        lr0=0.01,
        project=str(BASE_DIR / "runs"),
        name="lab_mouse_detector",
        exist_ok=True,
        verbose=True,
        plots=True
    )

    print("\n" + "=" * 60)
    print("[AI Training Complete] Finished fine-tuning lab model!")
    print(f" - Best weights saved at: {BASE_DIR}/runs/lab_mouse_detector/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    train()
