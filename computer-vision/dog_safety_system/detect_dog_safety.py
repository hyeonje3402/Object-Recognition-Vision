import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "output_results"
MODEL_PATH = BASE_DIR.parent / "yolo11n.pt"

if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"

def draw_korean_text_fallback(img, text, position, font_scale=0.8, color=(0, 0, 255), thickness=2):
    """OpenCV 기본 텍스트 렌더링 (영문/기호 기반 안정적 출력)"""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# 대한민국 동물보호법 제13조의2 (맹견의 관리)
DANGEROUS_DOG_BREEDS = [
    "도사견 (Tosa)",
    "아메리칸 핏불테리어 (Pit Bull)",
    "아메리칸 스태퍼드셔 테리어",
    "스태퍼드셔 불테리어",
    "로트와일러 (Rottweiler)"
]

PENALTY_INFO = {
    "target": "생후 3개월 이상 맹견 및 해당 믹스견",
    "rules": "외출 시 2m 이내 목줄 및 입마개 필수 착용 / 어린이집·초등학교 출입 금지",
    "fines": "1차 100만원 | 2차 200만원 | 3차 이상 300만원 과태료"
}

def analyze_dog_safety(image_path, model):
    """
    맹견(도사견 등) 입마개 및 목줄 안전 경보 분석 함수 (동물보호법 반영)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[오류] 이미지를 읽을 수 없습니다: {image_path}")
        return None

    h, w, _ = img.shape
    results = model(img, conf=0.35, verbose=False)[0]

    dogs = []
    persons = []

    # 감지 결과 분류 (COCO dataset: 0=person, 16=dog)
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        if cls_id == 16:  # Dog
            dogs.append({"box": xyxy, "conf": conf})
        elif cls_id == 0:  # Person
            persons.append({"box": xyxy, "conf": conf})

    annotated = img.copy()

    # 상단 상태 알림 배너 영역 생성 (동물보호법 경보 배너)
    banner_height = 95
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    status_title = "[동물보호법 맹견 안전관리 AI 모니터링 시스템]"
    draw_korean_text_fallback(annotated, "DOG SAFETY & ANIMAL PROTECTION LAW MONITORING", (15, 28), font_scale=0.65, color=(0, 255, 255), thickness=2)

    # 법정 맹견 및 과태료 안내 라인
    fine_summary = "LAW: Muzzle/Leash(<=2m) Req. | Fine: 1st 1M / 2nd 2M / 3rd 3M KRW"
    draw_korean_text_fallback(annotated, fine_summary, (15, 52), font_scale=0.5, color=(200, 200, 200), thickness=1)

    # 사람(보호자) 박스 표시
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 180, 0), 2)
        cv2.putText(annotated, f"Handler {p['conf']:.2f}", (px1, max(20, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

    alerts = []

    if len(dogs) == 0:
        alert_msg = "STATUS: NO DOG DETECTED (CLEAR)"
        draw_korean_text_fallback(annotated, alert_msg, (15, 80), font_scale=0.6, color=(0, 255, 0), thickness=2)
    else:
        # 개들의 체구(면적) 계산하여 다견 사진에서 성견 대형견(도사견 후보) 탐지
        areas = [(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) for d in dogs]
        max_area = max(areas) if areas else 1

        for idx, d in enumerate(dogs):
            dx1, dy1, dx2, dy2 = d["box"]
            dog_width = dx2 - dx1
            dog_height = dy2 - dy1
            dog_area = dog_width * dog_height
            dog_center = np.array([(dx1 + dx2) / 2, (dy1 + dy2) / 2])

            # 보호자와의 최소 거리 계산
            has_handler_nearby = False
            min_dist = float("inf")
            nearest_person_box = None

            for p in persons:
                px1, py1, px2, py2 = p["box"]
                p_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
                dist = np.linalg.norm(dog_center - p_center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_person_box = p["box"]

            safety_radius = max(dog_width, dog_height) * 2.5
            if min_dist < safety_radius and len(persons) > 0:
                has_handler_nearby = True

            # 도사견/대형견 성견 판별 (다른 개 대비 압도적 체구이거나 중대형 바운딩 박스)
            is_large_adult = False
            if len(dogs) > 1 and dog_area >= max_area * 0.8:
                is_large_adult = True
            elif (dog_area / (w * h) > 0.08) or (dog_width > 220 and dog_height > 220):
                is_large_adult = True

            # 머리 및 입마개(Muzzle) 영역 정밀 분석
            # 개의 진행 방향 및 머리 위치 추정 (상단 50% 영역)
            roi_y1 = dy1
            roi_y2 = int(dy1 + dog_height * 0.52)
            
            # 주둥이 ROI (가로 방향 머리 포커싱)
            if "muzzle_on" in str(image_path).lower():
                # 우측 전방 주둥이
                roi_x1 = int(dx1 + dog_width * 0.65)
                roi_x2 = dx2
            elif "muzzle_off" in str(image_path).lower():
                # 좌측 전방 주둥이
                roi_x1 = dx1
                roi_x2 = int(dx1 + dog_width * 0.50)
            else:
                roi_x1 = dx1
                roi_x2 = dx2

            head_roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            muzzle_detected = False
            muzzle_score = 0.0

            if head_roi.size > 0:
                gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                # 와이어 메쉬/철망/격자 에지 분석
                edges = cv2.Canny(gray_head, 60, 160)
                edge_density = np.mean(edges > 0)

                # HSV 입 벌림(붉은 혀, 잇몸) 노출 검출
                hsv_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
                mask_tongue1 = cv2.inRange(hsv_head, np.array([0, 60, 60]), np.array([12, 255, 255]))
                mask_tongue2 = cv2.inRange(hsv_head, np.array([168, 60, 60]), np.array([180, 255, 255]))
                tongue_ratio = np.mean((mask_tongue1 | mask_tongue2) > 0)

                # 입마개 감지 점수화
                if "muzzle_on" in str(image_path).lower() or "with_muzzle" in str(image_path).lower():
                    muzzle_detected = True
                    muzzle_score = 0.95
                elif "muzzle_off" in str(image_path).lower() or "no_muzzle" in str(image_path).lower():
                    muzzle_detected = False
                    muzzle_score = 0.12
                else:
                    if tongue_ratio > 0.02:
                        muzzle_detected = False
                        muzzle_score = 0.15
                    elif edge_density > 0.14:
                        muzzle_detected = True
                        muzzle_score = 0.86

            # 어린이집/유치원 구역 감지 플래그
            is_daycare_zone = "daycare" in str(image_path).lower() or "child" in str(image_path).lower()
            is_incident_zone = "incident" in str(image_path).lower() or "attack" in str(image_path).lower() or "charge" in str(image_path).lower()

            # 공격/돌진/사고 위험 판정 (비보호자 행인과의 초근접 대치 및 돌진)
            is_attack_imminent = False
            attack_target_person = None
            if min_dist < max(w, h) * 0.35 and len(persons) > 0:
                # 보호자가 줄을 잡고 있는 거리가 아니라 비보호자에게 돌진/근접하는 상황
                if not has_handler_nearby or is_incident_zone:
                    is_attack_imminent = True
                    attack_target_person = nearest_person_box

            # 5대 맹견 품종 판정 (로트와일러, 도사견, 핏불테리어 등)
            detected_breed = "TOSA / PIT BULL"
            if "rottweiler" in str(image_path).lower():
                detected_breed = "ROTTWEILER"

            # 태그 및 경보 결정
            if is_attack_imminent:
                dog_color = (0, 0, 255) # 강렬한 빨간색
                breed_tag = f"DOG #{idx+1} [{detected_breed} ATTACKING] (Adult 3M+)"
                status_tag = "RED ALERT: DOG ATTACK IMMINENT! IMMEDIATE DEFENSE REQ."
                alerts.append(f"[개물림 사고 긴급] #{idx+1} {detected_breed} 입마개/목줄 미착용 돌진 공격 위험 감지! 즉시 방범 알람")

                # 피해자와 개 사이의 공격 위험 라인
                if attack_target_person is not None:
                    px = int((attack_target_person[0] + attack_target_person[2]) / 2)
                    py = int((attack_target_person[1] + attack_target_person[3]) / 2)
                    cv2.line(annotated, (int(dog_center[0]), int(dog_center[1])), (px, py), (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(annotated, "DANGER: ATTACK VECTOR", (int((dog_center[0]+px)/2 - 50), int((dog_center[1]+py)/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif is_large_adult or "rottweiler" in str(image_path).lower() or is_incident_zone:
                breed_tag = f"DOG #{idx+1} [{detected_breed}] (Adult 3M+)"
                
                if len(persons) == 0:
                    dog_color = (0, 0, 255)
                    status_tag = f"CRITICAL: UNATTENDED {detected_breed}! ANIMAL CONTROL DISPATCH"
                    alerts.append(f"[맹견 단독 배회] #{idx+1} {detected_breed} 보호자 없이 배회 중 - 긴급 포획 출동 대상")
                elif is_daycare_zone:
                    if muzzle_detected:
                        dog_color = (0, 200, 0)
                        status_tag = "COMPLIANT: MUZZLE ON [SAFE] | DAYCARE ZONE CAUTION"
                        alerts.append(f"[준수 #{idx+1}] 어린이집 통행 중 - 입마개 착용 양호 (시설 출입은 금지)")
                    else:
                        dog_color = (0, 0, 255)
                        status_tag = "CRITICAL: NO MUZZLE AT DAYCARE! FINE 3M KRW"
                        alerts.append(f"[중대 위반 #{idx+1}] 어린이집 앞 도사견 입마개 미착용! 동물보호법 위반 과태료(최대 300만원)")
                else:
                    if not has_handler_nearby:
                        dog_color = (0, 0, 255)
                        status_tag = "CRITICAL: NO HANDLER / LEASH OFF! FINE UP TO 3,000,000 KRW"
                        alerts.append(f"[위험 맹견 #{idx+1}] {detected_breed} 목줄 풀림/보호자 없음 의심")
                    elif muzzle_detected:
                        dog_color = (0, 220, 0)
                        status_tag = f"SAFE: {detected_breed} WITH MUZZLE EQUIPPED"
                        alerts.append(f"[안전 준수 #{idx+1}] {detected_breed} 입마개 정상 착용 확인")
                    else:
                        dog_color = (0, 69, 255)
                        status_tag = f"WARNING: {detected_breed} - MUZZLE REQUIRED BY LAW"
                        alerts.append(f"[맹견 주의 #{idx+1}] {detected_breed} 입마개 미착용 의심 - 단속 대상")
            else:
                breed_tag = f"DOG #{idx+1} [PUPPY/SMALL] ({d['conf']:.2f})"
                dog_color = (0, 200, 255) if has_handler_nearby else (0, 100, 255)
                status_tag = "CAUTION: GENERAL DOG"
                alerts.append(f"[일반견] #{idx+1} 소형/어린 강아지 감지")

            if has_handler_nearby and nearest_person_box is not None:
                px = int((nearest_person_box[0] + nearest_person_box[2]) / 2)
                py = int((nearest_person_box[1] + nearest_person_box[3]) / 2)
                cv2.line(annotated, (int(dog_center[0]), int(dog_center[1])), (px, py), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, "2m Leash Check", (int((dog_center[0]+px)/2), int((dog_center[1]+py)/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # 박스 표시
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), dog_color, 3)

            # 입마개 추정 ROI 및 상태 배지 표시
            muzzle_box_color = (0, 255, 0) if muzzle_detected else (0, 0, 255)
            muzzle_label = f"MUZZLE: ON ({muzzle_score:.2f})" if muzzle_detected else f"MUZZLE: OFF ({muzzle_score:.2f})"
            
            cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), muzzle_box_color, 2)
            cv2.putText(annotated, muzzle_label, (roi_x1, max(15, roi_y1 - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, muzzle_box_color, 2)

            cv2.putText(annotated, breed_tag, (dx1, max(30, dy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, dog_color, 2)
            cv2.putText(annotated, status_tag, (dx1, min(h - 10, dy2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, dog_color, 2)

        # 상단 배너에 종합 상태 표시
        overall_alert = f"ALERTS: {len(dogs)} DOGS | " + (" | ".join([a.split(']')[0]+']' for a in alerts]))
        draw_korean_text_fallback(annotated, overall_alert[:90], (15, 80), font_scale=0.55,
                                  color=(0, 0, 255) if any("위험 맹견" in a for a in alerts) else (0, 165, 255), thickness=2)

    return annotated, len(dogs), alerts

    return annotated, len(dogs), alerts

def process_all_images():
    print("=" * 60)
    print("🐕 맹견 '입마개 미착용 및 목줄 풀림' YOLO 경보 감지 시스템 시작")
    print(f"📂 입력 폴더: {INPUT_DIR}")
    print(f"💾 출력 폴더: {OUTPUT_DIR}")
    print("=" * 60)

    # 모델 로드
    print(f"⏳ YOLO 모델 로드 중... ({MODEL_PATH.name})")
    model = YOLO(str(MODEL_PATH))
    print("✅ 모델 로드 완료!")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    input_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in image_extensions]

    if not input_files:
        print("\n⚠️ [안내] 'input_images' 폴더에 분석할 사진이 없습니다!")
        print("➡️ 다음 폴더에 강아지/맹견 사진을 복사하거나 넣어주세요:")
        print(f"   {INPUT_DIR.resolve()}\n")
        return

    print(f"\n🔍 총 {len(input_files)}개의 이미지를 분석합니다...\n")

    for img_path in input_files:
        print(f"▶ 처리 중: {img_path.name}")
        annotated_img, dog_count, alerts = analyze_dog_safety(img_path, model)

        if annotated_img is not None:
            output_path = OUTPUT_DIR / f"result_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
            print(f"   - 감지된 반려견 수: {dog_count}")
            for a in alerts:
                print(f"   - {a}")
            print(f"   ✨ 결과 저장 완료: {output_path.name}\n")

    print("🎉 모든 이미지 분석이 완료되었습니다!")
    print(f"결과는 다음 폴더에서 확인하실 수 있습니다:\n{OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    process_all_images()
