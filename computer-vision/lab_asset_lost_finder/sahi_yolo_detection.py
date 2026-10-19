"""
[SAHI + Ultralytics YOLO 초소형 객체 탐지 파이프라인]
- 고해상도(예: 2944x1248, 4K 등) CCTV 하이앵글 영상/이미지 특화
- SAHI(Slicing Aided Hyper Inference)를 통한 작은 전자기기(노트북, 태블릿/휴대폰, 마우스) 검출 극대화
"""

import os
import cv2
import numpy as np
from pathlib import Path

# SAHI 핵심 모듈 임포트
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
except ImportError:
    raise ImportError(
        "SAHI 라이브러리가 설치되어 있지 않습니다.\n"
        "터미널에서 'pip install sahi' 를 실행해 주세요."
    )


def run_sahi_small_object_detection(
    image_path: str,
    output_path: str = "sahi_detection_result.jpg",
    model_path: str = "yolov8x.pt",  # 또는 yolo11x.pt 등 최신 고성능 모델
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    conf_threshold: float = 0.25,
    device: str = "cuda:0"  # GPU가 없을 경우 "cpu"
):
    """
    고해상도 이미지에서 SAHI 기반 슬라이싱 추론을 수행하고 
    노트북, 태블릿/스마트폰, 마우스 클래스만 필터링하여 시각화합니다.
    """
    print(f"🚀 [1/4] 모델 로드 중... ({model_path} on {device})")
    
    # 1. SAHI용 YOLO 감지 모델 래퍼 초기화
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",          # Ultralytics v8 및 v11 호환
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )

    print(f"📷 [2/4] 이미지 로드 중: {image_path}")
    image = read_image(image_path)
    img_h, img_w = image.shape[:2]
    print(f"    - 이미지 해상도: {img_w}x{img_h}")

    print(f"🔍 [3/4] SAHI 슬라이싱 추론 실행 중 (슬라이스: {slice_size}x{slice_size}, 오버랩: {overlap_ratio})")
    
    # 2. SAHI 슬라이싱 하이퍼 추론 실행
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=True,      # 전체 축소 이미지 추론도 병합하여 큰 객체도 놓치지 않음
        postprocess_type="GREEDYNMM",    # 슬라이스 경계면 중복 제거 알고리즘 (NMS or GREEDYNMM)
        postprocess_match_metric="IOS",  # Intersection over Smaller Area (작은 객체에 유리)
        postprocess_match_threshold=0.5, # 중복 병합 임계값
        verbose=1
    )

    # 3. 목표 클래스 정의 및 색상 팔레트 (BGR 포맷)
    TARGET_CLASSES = {
        "laptop": {"name_ko": "노트북", "color": (255, 128, 0)},       # 주황/스카이블루 계열
        "cell phone": {"name_ko": "태블릿/폰", "color": (0, 215, 255)}, # 밝은 노랑
        "mouse": {"name_ko": "마우스", "color": (50, 255, 50)}          # 연두색
    }

    # 원본 이미지 복사 (시각화용)
    annotated_image = image.copy()
    # OpenCV 컬러 스페이스 호환을 위해 RGB -> BGR 변환 (필요시)
    if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    print("🎨 [4/4] 결과 필터링 및 시각화 진행...")
    detected_counts = {k: 0 for k in TARGET_CLASSES}
    total_filtered_boxes = 0

    # 4. 검출된 객체 목록 순회 및 필터링
    for obj_pred in result.object_prediction_list:
        category_name = obj_pred.category.name.lower()
        score = obj_pred.score.value
        
        # 관심 대상 클래스인지 확인
        if category_name in TARGET_CLASSES:
            detected_counts[category_name] += 1
            total_filtered_boxes += 1
            
            # 바운딩 박스 정수 좌표 변환 [x1, y1, x2, y2]
            bbox = obj_pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            
            # 클래스별 색상 및 라벨 가져오기
            cls_info = TARGET_CLASSES[category_name]
            color = cls_info["color"]
            label_text = f"{cls_info['name_ko']} {score:.2f}"
            
            # 바운딩 박스 그리기 (두께 2px)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트 배경 박스 (가독성 향상)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # 텍스트 박스 위치 조정 (화면 밖으로 나가지 않도록)
            text_y1 = max(y1 - text_h - 6, 0)
            text_y2 = text_y1 + text_h + 6
            text_x2 = min(x1 + text_w + 6, img_w)
            
            cv2.rectangle(annotated_image, (x1, text_y1), (text_x2, text_y2), color, -1)
            cv2.putText(
                annotated_image,
                label_text,
                (x1 + 3, text_y2 - 4),
                font,
                font_scale,
                (0, 0, 0),  # 검은색 텍스트
                font_thickness,
                lineType=cv2.LINE_AA
            )

    # 5. 상단 정보 오버레이 배너
    header_text = (
        f"SAHI Detection Result | "
        f"Laptop: {detected_counts['laptop']} | "
        f"Tablet/Phone: {detected_counts['cell phone']} | "
        f"Mouse: {detected_counts['mouse']}"
    )
    cv2.rectangle(annotated_image, (0, 0), (img_w, 35), (20, 20, 20), -1)
    cv2.putText(
        annotated_image,
        header_text,
        (15, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA
    )

    # 6. 최종 이미지 저장
    cv2.imwrite(output_path, annotated_image)
    print(f"\n✅ 검출 완료! 최종 결과가 저장되었습니다: {output_path}")
    print(f"📊 검출 통계: 노트북 {detected_counts['laptop']}개, 태블릿/폰 {detected_counts['cell phone']}개, 마우스 {detected_counts['mouse']}개 (총 {total_filtered_boxes}개)")
    
    return annotated_image, detected_counts


if __name__ == "__main__":
    # 프로젝트 내 테스트 이미지 경로 자동 탐색 예시
    current_dir = Path(__file__).resolve().parent
    sample_image = current_dir / "check_new_video2_thumb.jpg"
    
    # 썸네일 또는 고해상도 샘플 이미지가 있을 경우 실행
    if sample_image.exists():
        target_img = str(sample_image)
    else:
        # 기본 대체 이미지
        target_img = str(current_dir / "sample_frame_0.jpg")
    
    output_file = str(current_dir / "output_results" / "sahi_detected_result.jpg")
    os.makedirs(str(current_dir / "output_results"), exist_ok=True)
    
    # GPU 사용 가능 여부 체크
    import torch
    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"실행 대상 이미지: {target_img}")
    run_sahi_small_object_detection(
        image_path=target_img,
        output_path=output_file,
        model_path="yolov8x.pt",      # 첫 실행 시 자동으로 가중치가 다운로드됩니다.
        slice_size=640,
        overlap_ratio=0.2,
        conf_threshold=0.25,
        device=device_to_use
    )
