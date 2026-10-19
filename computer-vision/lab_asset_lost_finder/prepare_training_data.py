"""
[고도화된 실습실 학습 데이터 생성기 - prepare_training_data.py]
실습실 CCTV에서 마우스/소형 물품의 해상도 손실 문제를 해결하기 위해:
1. 전체 프레임 조명/대비/색조/가우시안 변형 증강 (12장)
2. 주요 좌석 영역 고해상도 로컬 패치 크롭 및 좌표 리매핑 (8장)
총 20장 이상의 고품질 데이터셋을 생성하여 YOLO가 작은 마우스의 형태적 특징을 명확히 학습하도록 함.
"""
import os
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMG_SRC = BASE_DIR / "input_images" / "lab_classroom_cctv.jpg"

def create_dataset():
    print("[Dataset Generation] Initializing high-precision lab equipment dataset...")
    img = cv2.imread(str(IMG_SRC))
    if img is None:
        print("[Error] Source image not found!")
        return

    h, w, _ = img.shape

    # 1. 원본 이미지 내 마우스 및 기자재 전수 라벨링 (x1, y1, x2, y2)
    # 0: mouse, 1: public_mouse, 2: keyboard, 3: small_item
    ground_truth_boxes = [
        # 교탁 공용 마우스 (class 1: public_mouse)
        (1, 332, 713, 365, 739),
        # 교탁 공용 키보드 (class 2: keyboard)
        (2, 255, 647, 356, 720),
        # 1번 학생 마우스 (class 0: mouse)
        (0, 525, 600, 548, 622),
        # 1번 학생 키보드 (class 2)
        (2, 470, 580, 520, 610),
        # 1번 학생 볼펜 (class 3: small_item)
        (3, 530, 625, 555, 635),
        # 2번 학생 마우스 (class 0)
        (0, 655, 665, 682, 692),
        # 2번 학생 키보드 (class 2)
        (2, 580, 635, 645, 670),
        # 2번 학생 볼펜 (class 3)
        (3, 640, 685, 665, 695),
        # 3번 학생 마우스 (class 0)
        (0, 735, 660, 762, 687),
        # 3번 학생 키보드 (class 2)
        (2, 670, 635, 730, 670),
        # 3번 학생 에어팟 (class 3)
        (3, 715, 630, 735, 648),
        # 4번 좌석 마우스 (class 0)
        (0, 505, 520, 526, 540),
        # 5번 좌석 마우스 (class 0)
        (0, 615, 540, 636, 560),
        # 6번 좌석 마우스 (class 0)
        (0, 710, 535, 732, 557),
    ]

    def to_yolo_format(box, img_w, img_h):
        cls_id, x1, y1, x2, y2 = box
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        bw = (x2 - x1) / float(img_w)
        bh = (y2 - y1) / float(img_h)
        return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

    # 디렉터리 준비
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. 전체 프레임 증강 세트
    full_transforms = [
        ("orig", lambda im: im),
        ("bright", lambda im: cv2.convertScaleAbs(im, alpha=1.15, beta=20)),
        ("dark", lambda im: cv2.convertScaleAbs(im, alpha=0.85, beta=-20)),
        ("contrast_high", lambda im: cv2.convertScaleAbs(im, alpha=1.3, beta=0)),
        ("contrast_low", lambda im: cv2.convertScaleAbs(im, alpha=0.8, beta=10)),
        ("clahe", lambda im: cv2.cvtColor(cv2.merge([
            cv2.createCLAHE(3.0, (8,8)).apply(cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))[0]),
            cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))[1],
            cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))[2]
        ]), cv2.COLOR_LAB2BGR)),
        ("blur", lambda im: cv2.GaussianBlur(im, (3, 3), 0.8)),
        ("hsv_shift", lambda im: cv2.cvtColor(cv2.add(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), np.array([5, 10, 5], dtype=np.uint8)), cv2.COLOR_HSV2BGR)),
    ]

    train_count = 0
    val_count = 0

    # 전체 이미지 저장 (일부는 train, 일부는 val)
    for idx, (name, fn) in enumerate(full_transforms):
        aug_im = fn(img.copy())
        split = "val" if idx in [2, 6] else "train"
        if split == "train":
            train_count += 1
        else:
            val_count += 1

        img_path = DATASET_DIR / "images" / split / f"full_{name}.jpg"
        lbl_path = DATASET_DIR / "labels" / split / f"full_{name}.txt"
        cv2.imwrite(str(img_path), aug_im)

        with open(lbl_path, "w", encoding="utf-8") as f:
            for b in ground_truth_boxes:
                f.write(to_yolo_format(b, w, h))

    # 2. 책상 및 좌석 영역별 고해상도 로컬 크롭 패치 (작은 마우스 특징 극대화 학습)
    crop_regions = [
        ("podium", 150, 580, 480, 850),          # 교탁 영역 (공용 마우스, 키보드)
        ("front_seats", 420, 530, 800, 780),     # 앞줄 1~3번 좌석 (마우스 3개, 키보드 3개, 펜, 에어팟)
        ("mid_seats", 440, 480, 780, 620),       # 중간 4~6번 좌석 (마우스 3개, 키보드)
        ("front_desk_left", 440, 550, 680, 750), # 1, 2번 좌석 집중 크롭
        ("front_desk_right", 600, 580, 820, 760),# 2, 3번 좌석 집중 크롭
    ]

    for crop_name, cx1, cy1, cx2, cy2 in crop_regions:
        cropped_im = img[cy1:cy2, cx1:cx2].copy()
        cw = cx2 - cx1
        ch = cy2 - cy1

        # 크롭 영역에 포함된 박스들 좌표 재계산
        cropped_boxes = []
        for cls_id, bx1, by1, bx2, by2 in ground_truth_boxes:
            # 크롭 내부와 겹치는지 체크
            inter_x1 = max(bx1, cx1)
            inter_y1 = max(by1, cy1)
            inter_x2 = min(bx2, cx2)
            inter_y2 = min(by2, cy2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                # 크롭 내부 상대 좌표로 변환
                rel_x1 = inter_x1 - cx1
                rel_y1 = inter_y1 - cy1
                rel_x2 = inter_x2 - cx1
                rel_y2 = inter_y2 - cy1
                cropped_boxes.append((cls_id, rel_x1, rel_y1, rel_x2, rel_y2))

        if not cropped_boxes:
            continue

        # 원본 크롭 및 밝기 변형 크롭 생성
        for sub_name, sub_fn in [("base", lambda m: m), ("bright", lambda m: cv2.convertScaleAbs(m, alpha=1.1, beta=15))]:
            aug_crop = sub_fn(cropped_im.copy())
            split = "train"
            train_count += 1

            img_path = DATASET_DIR / "images" / split / f"crop_{crop_name}_{sub_name}.jpg"
            lbl_path = DATASET_DIR / "labels" / split / f"crop_{crop_name}_{sub_name}.txt"
            cv2.imwrite(str(img_path), aug_crop)

            with open(lbl_path, "w", encoding="utf-8") as f:
                for b in cropped_boxes:
                    f.write(to_yolo_format(b, cw, ch))

    # 3. data.yaml 갱신
    yaml_content = f"""# 실습실 기자재 및 마우스 전용 고정밀 커스텀 데이터셋 설정
path: {DATASET_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: mouse
  1: public_mouse
  2: keyboard
  3: small_item
"""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[Dataset] Generated {train_count} train images and {val_count} val images.")
    print(f"[Dataset] Configuration saved to {yaml_path}")

if __name__ == "__main__":
    create_dataset()
