"""
[실습실 초소형 분실물 및 기자재 커스텀 데이터셋 생성기 - prepare_training_data.py]
- 클래스 확장:
  0: mouse (일반 개인 마우스)
  1: public_mouse (교탁 공용 마우스 - 도난 모니터링)
  2: keyboard (키보드)
  3: airpods (에어팟 / 무선 이어폰 케이스 - 대학생 빈출 초소형 분실물)
  4: charger (스마트폰/노트북 충전 어댑터 & 케이블 - 좌석 방치 빈출 분실물)
- 전체 프레임 증강 (밝기, 어두움, 대비, CLAHE, 가우시안 블러, 노이즈)
- 책상 영역 고해상도 로컬 패치 크롭 (SAHI 및 소형 객체 특징 극대화)
- Copy-Paste 기법을 통한 현실적인 에어팟 및 충전기 배치 샘플 생성
"""
import os
import random
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMG_SRC = BASE_DIR / "input_images" / "lab_classroom_cctv.jpg"

def draw_realistic_airpods(im, x, y, w, h):
    """실습실 책상 위에 놓인 에어팟 무선 이어폰 케이스 합성 렌더링"""
    # 은은한 그림자
    cv2.ellipse(im, (x + w // 2 + 2, y + h // 2 + 3), (w // 2 + 3, h // 2 + 2), 0, 0, 360, (50, 50, 50), -1)
    # 흰색 둥근 본체 케이스
    cv2.ellipse(im, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (230, 235, 238), -1)
    # 상단 뚜껑 분할선(힌지)
    cv2.line(im, (x + 3, y + h // 3), (x + w - 3, y + h // 3), (170, 175, 180), 1)
    # 중앙 상태 표시 LED 라이트 (녹색 또는 앰버)
    led_color = (0, 230, 100) if random.random() > 0.3 else (0, 165, 255)
    cv2.circle(im, (x + w // 2, y + h * 2 // 3), 1, led_color, -1)

def draw_realistic_charger(im, x, y, w, h):
    """실습실 책상/콘센트 부근 충전 어댑터 및 연결 케이블 합성 렌더링"""
    # 그림자
    cv2.rectangle(im, (x + 2, y + 2), (x + w + 2, y + h + 2), (40, 40, 40), -1)
    # 어댑터 본체 (화이트/블랙 큐브)
    body_color = (220, 225, 230) if random.random() > 0.4 else (45, 45, 50)
    cv2.rectangle(im, (x, y), (x + w, y + h), body_color, -1)
    cv2.rectangle(im, (x, y), (x + w, y + h), (120, 120, 120), 1)
    # C타입 단자 포트
    cv2.rectangle(im, (x + w // 4, y + h - 2), (x + w * 3 // 4, y + h), (90, 90, 90), -1)
    # 케이블 라인 (책상 위로 곡선 형태)
    curve_pts = np.array([
        [x + w // 2, y + h],
        [x + w // 2 + random.randint(-10, 10), y + h + 15],
        [x + w // 2 + random.randint(10, 35), y + h + 30]
    ], np.int32)
    cv2.polylines(im, [curve_pts], False, (190, 195, 200), 2)

def create_dataset():
    print("[Dataset Generation] Initializing lab equipment & personal lost items dataset...")
    img = cv2.imread(str(IMG_SRC))
    if img is None:
        print("[Error] Source image not found at:", IMG_SRC)
        return

    h, w, _ = img.shape

    # 기준 객체 라벨링 (cls_id, x1, y1, x2, y2)
    # 0: mouse, 1: public_mouse, 2: keyboard, 3: airpods, 4: charger
    ground_truth_boxes = [
        # 1. 교탁 공용 마우스 (class 1: public_mouse)
        (1, 278, 796, 305, 828),
        # 2. 교탁 공용 키보드 (class 2: keyboard)
        (2, 215, 720, 300, 808),
        # 3. 교탁 위 충전 어댑터 (class 4: charger)
        (4, 345, 715, 375, 745),

        # 4. 1번 학생 좌석 마우스 (class 0: mouse)
        (0, 525, 600, 548, 622),
        # 5. 1번 학생 키보드 (class 2: keyboard)
        (2, 470, 580, 520, 610),
        # 6. 1번 학생 책상 충전기 어댑터 (class 4: charger)
        (4, 540, 570, 568, 598),

        # 7. 2번 학생 좌석 마우스 (class 0: mouse)
        (0, 655, 665, 682, 692),
        # 8. 2번 학생 키보드 (class 2: keyboard)
        (2, 580, 635, 645, 670),

        # 9. 3번 학생 좌석 마우스 (class 0: mouse)
        (0, 735, 660, 762, 687),
        # 10. 3번 학생 키보드 (class 2: keyboard)
        (2, 670, 635, 730, 670),
        # 11. 3번 학생 책상 위 에어팟 케이스 (class 3: airpods)
        (3, 715, 630, 736, 650),

        # 12. 4번 좌석 마우스 (class 0: mouse)
        (0, 505, 520, 526, 540),
        # 13. 5번 좌석 마우스 (class 0: mouse)
        (0, 615, 540, 636, 560),
        # 14. 6번 좌석 마우스 (class 0: mouse)
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

    # 기본 증강 세트
    full_transforms = [
        ("orig", lambda im: im),
        ("bright", lambda im: cv2.convertScaleAbs(im, alpha=1.15, beta=18)),
        ("dark_recess", lambda im: cv2.convertScaleAbs(im, alpha=0.82, beta=-22)),
        ("contrast_high", lambda im: cv2.convertScaleAbs(im, alpha=1.28, beta=0)),
        ("contrast_low", lambda im: cv2.convertScaleAbs(im, alpha=0.85, beta=10)),
        ("clahe", lambda im: cv2.cvtColor(cv2.merge([
            cv2.createCLAHE(3.0, (8, 8)).apply(cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))[0]),
            cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))[1],
            cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))[2]
        ]), cv2.COLOR_LAB2BGR)),
        ("blur", lambda im: cv2.GaussianBlur(im, (3, 3), 0.8)),
        ("hsv_shift", lambda im: cv2.cvtColor(cv2.add(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), np.array([4, 8, 4], dtype=np.uint8)), cv2.COLOR_HSV2BGR)),
    ]

    train_count = 0
    val_count = 0

    # 1. 전체 이미지 저장
    for idx, (name, fn) in enumerate(full_transforms):
        aug_im = fn(img.copy())
        current_boxes = list(ground_truth_boxes)

        # 에어팟 및 충전기 객체를 책상 위 빈 공간에 자연스럽게 렌더링
        draw_realistic_airpods(aug_im, 715, 630, 21, 20)
        draw_realistic_charger(aug_im, 540, 570, 28, 28)
        draw_realistic_charger(aug_im, 345, 715, 30, 30)

        # 추가 변형: 2번 학생 좌석에도 에어팟/충전기 추가
        if idx in [1, 3, 5, 7]:
            draw_realistic_airpods(aug_im, 642, 680, 22, 21)
            current_boxes.append((3, 642, 680, 664, 701))
            draw_realistic_charger(aug_im, 490, 620, 26, 26)
            current_boxes.append((4, 490, 620, 516, 646))

        split = "val" if idx in [2, 5] else "train"
        if split == "train":
            train_count += 1
        else:
            val_count += 1

        img_path = DATASET_DIR / "images" / split / f"full_{name}.jpg"
        lbl_path = DATASET_DIR / "labels" / split / f"full_{name}.txt"
        cv2.imwrite(str(img_path), aug_im)

        with open(lbl_path, "w", encoding="utf-8") as f:
            for b in current_boxes:
                f.write(to_yolo_format(b, w, h))

    # 2. 책상 및 좌석 영역별 고해상도 로컬 크롭 패치 (SAHI 및 소형 객체 특징 극대화)
    crop_regions = [
        ("podium", 180, 690, 420, 850),          # 교탁 영역 (공용 마우스, 키보드, 충전기)
        ("seat_1_desk", 450, 550, 580, 640),     # 1번 좌석 책상 (마우스, 키보드, 충전기)
        ("seat_2_desk", 570, 610, 700, 710),     # 2번 좌석 책상 (마우스, 키보드, 에어팟)
        ("seat_3_desk", 660, 610, 780, 710),     # 3번 좌석 책상 (마우스, 키보드, 에어팟)
        ("front_seats", 430, 540, 790, 730),     # 1~3번 앞줄 좌석 전체 패치
        ("mid_seats", 480, 500, 750, 600),       # 4~6번 중간 좌석 전체 패치
    ]

    base_patch_boxes = list(ground_truth_boxes)
    # 크롭용 기준 이미지에도 물품 합성
    crop_base_im = img.copy()
    draw_realistic_airpods(crop_base_im, 715, 630, 21, 20)
    draw_realistic_charger(crop_base_im, 540, 570, 28, 28)
    draw_realistic_charger(crop_base_im, 345, 715, 30, 30)

    for crop_name, cx1, cy1, cx2, cy2 in crop_regions:
        cropped_im = crop_base_im[cy1:cy2, cx1:cx2].copy()
        cw = cx2 - cx1
        ch = cy2 - cy1

        cropped_boxes = []
        for cls_id, bx1, by1, bx2, by2 in base_patch_boxes:
            inter_x1 = max(bx1, cx1)
            inter_y1 = max(by1, cy1)
            inter_x2 = min(bx2, cx2)
            inter_y2 = min(by2, cy2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                rel_x1 = inter_x1 - cx1
                rel_y1 = inter_y1 - cy1
                rel_x2 = inter_x2 - cx1
                rel_y2 = inter_y2 - cy1
                cropped_boxes.append((cls_id, rel_x1, rel_y1, rel_x2, rel_y2))

        if not cropped_boxes:
            continue

        for sub_name, sub_fn in [
            ("base", lambda m: m),
            ("bright", lambda m: cv2.convertScaleAbs(m, alpha=1.12, beta=15)),
            ("clahe", lambda m: cv2.cvtColor(cv2.merge([
                cv2.createCLAHE(2.5, (8, 8)).apply(cv2.split(cv2.cvtColor(m, cv2.COLOR_BGR2LAB))[0]),
                cv2.split(cv2.cvtColor(m, cv2.COLOR_BGR2LAB))[1],
                cv2.split(cv2.cvtColor(m, cv2.COLOR_BGR2LAB))[2]
            ]), cv2.COLOR_LAB2BGR))
        ]:
            aug_crop = sub_fn(cropped_im.copy())
            split = "train"
            train_count += 1

            img_path = DATASET_DIR / "images" / split / f"crop_{crop_name}_{sub_name}.jpg"
            lbl_path = DATASET_DIR / "labels" / split / f"crop_{crop_name}_{sub_name}.txt"
            cv2.imwrite(str(img_path), aug_crop)

            with open(lbl_path, "w", encoding="utf-8") as f:
                for b in cropped_boxes:
                    f.write(to_yolo_format(b, cw, ch))

    # 3. data.yaml 생성
    yaml_content = f"""# 실습실 기자재 & 초소형 개인 분실물(에어팟, 충전기) 커스텀 데이터셋
path: {DATASET_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: mouse
  1: public_mouse
  2: keyboard
  3: airpods
  4: charger
"""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[Dataset Complete] Generated {train_count} train images and {val_count} val images.")
    print(f"[Dataset Config] Configuration saved to {yaml_path}")
    print("[Classes]")
    print("  0: mouse (개인 마우스)")
    print("  1: public_mouse (교탁 공용 마우스)")
    print("  2: keyboard (키보드)")
    print("  3: airpods (에어팟/무선 이어폰)")
    print("  4: charger (충전 어댑터/케이블)")

if __name__ == "__main__":
    create_dataset()
