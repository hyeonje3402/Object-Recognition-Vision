"""
[SAHI + Ultralytics YOLO 영상(Video) 초소형 객체 탐지 파이프라인]
- 고해상도 CCTV 동영상(mp4, avi 등)을 프레임 단위로 읽어 SAHI 슬라이싱 추론 적용
- 노트북(Laptop), 태블릿/스마트폰(Tablet/Phone), 마우스(Mouse) 정밀 검출
- cv2.VideoWriter를 통해 실시간 HUD 및 바운딩 박스가 합성된 고화질 결과 영상 생성
"""

import os
import time
import cv2
import numpy as np
import torch
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def annotate_frame_with_sahi(
    frame: np.ndarray,
    detection_model: AutoDetectionModel,
    target_classes: dict,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    frame_idx: int = 0,
    total_frames: int = 0
) -> np.ndarray:
    """
    단일 영상 프레임(BGR)에 SAHI 슬라이싱 추론을 수행하고 바운딩 박스 및 HUD를 그려 반환합니다.
    """
    img_h, img_w = frame.shape[:2]

    # SAHI 및 YOLO 모델은 RGB 이미지를 기준으로 사전 학습되었으므로 BGR -> RGB 변환 필수
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # SAHI 슬라이싱 추론 실행
    result = get_sliced_prediction(
        image=rgb_frame,
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=True,       # 전체 축소 화면 추론 병합 (큰 객체 보존)
        postprocess_type="GREEDYNMM",     # 경계면 NMM 중복 필터링
        postprocess_match_metric="IOS",   # 작은 객체 억제 방지 (Intersection over Smaller)
        postprocess_match_threshold=0.5,
        verbose=0                         # 비디오 처리 중 프레임별 내부 로그 출력 비활성화
    )

    annotated = frame.copy()
    detected_counts = {k: 0 for k in target_classes}

    # 탐지 결과 필터링 및 렌더링
    for obj_pred in result.object_prediction_list:
        cat_name = obj_pred.category.name.lower()
        score = obj_pred.score.value

        if cat_name in target_classes:
            detected_counts[cat_name] += 1
            x1, y1, x2, y2 = map(int, obj_pred.bbox.to_xyxy())

            # 좌표 범위 클리핑
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            cls_info = target_classes[cat_name]
            color = cls_info["color"]
            label_text = f"{cls_info['name_ko']} {score:.2f}"

            # 1. 바운딩 박스
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 2. 라벨 텍스트 박스
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            ty1 = max(y1 - th - 6, 0)
            ty2 = ty1 + th + 6
            tx2 = min(x1 + tw + 6, img_w)

            cv2.rectangle(annotated, (x1, ty1), (tx2, ty2), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, ty2 - 4),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                lineType=cv2.LINE_AA
            )

    # 3. 상단 HUD 상태 바 (대시보드)
    hud_h = 40
    hud_overlay = annotated[:hud_h, :].copy()
    cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (25, 25, 25), -1)
    cv2.addWeighted(hud_overlay, 0.3, annotated[:hud_h, :], 0.7, 0, annotated[:hud_h, :])

    progress_str = f"[{frame_idx}/{total_frames}]" if total_frames > 0 else f"[{frame_idx}]"
    hud_text = (
        f"SAHI CCTV AI {progress_str} | "
        f"Laptop: {detected_counts['laptop']} | "
        f"Tablet/Phone: {detected_counts['cell phone']} | "
        f"Mouse: {detected_counts['mouse']}"
    )
    cv2.putText(
        annotated,
        hud_text,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 200),
        2,
        lineType=cv2.LINE_AA
    )

    return annotated


def run_sahi_video_detection(
    input_video_path: str,
    output_video_path: str,
    model_path: str = "yolov8x.pt",       # 최신 고성능 모델 권장 (yolov8x.pt, yolo11x.pt 등)
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    conf_threshold: float = 0.25,
    max_frames: int = None,               # 테스트용 최대 프레임 수 (전체 처리 시 None)
    frame_stride: int = 1,                # 프레임 간격 (1이면 전수 처리, 2면 2프레임마다 1번 처리)
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    """
    입력 동영상을 프레임 단위로 읽어 SAHI 슬라이싱 검출을 적용하고
    최종 결과 동영상(.mp4)으로 저장합니다.
    """
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 영상 파일을 찾을 수 없습니다: {input_video_path}")

    output_dir = Path(output_video_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🎬 [SAHI Video Detection Pipeline]")
    print(f" • 입력 영상: {input_path.name}")
    print(f" • 출력 경로: {output_video_path}")
    print(f" • 연산 장치: {device}")
    print(f" • 탐지 모델: {model_path} (Confidence: {conf_threshold})")
    print("=" * 70)

    # 1. SAHI YOLO 모델 초기화
    print("🚀 [1/3] SAHI 모델 가중치 로드 중...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )

    # 2. VideoCapture 초기화 및 메타데이터 추출
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None:
        target_total = min(total_frames, max_frames)
    else:
        target_total = total_frames

    print(f"📹 [2/3] 비디오 스펙: {width}x{height} @ {fps:.2f}fps (총 {total_frames} 프레임)")

    # 3. VideoWriter 초기화 (mp4v 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / frame_stride
    out = cv2.VideoWriter(str(output_video_path), fourcc, out_fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"VideoWriter를 초기화할 수 없습니다: {output_video_path}")

    # COCO 기준 탐지 대상 및 색상 팔레트
    target_classes = {
        "laptop": {"name_ko": "노트북", "color": (255, 128, 0)},       # 스카이블루/오렌지
        "cell phone": {"name_ko": "태블릿/폰", "color": (0, 215, 255)}, # 밝은 노랑
        "mouse": {"name_ko": "마우스", "color": (50, 255, 50)}          # 연두색
    }

    print(f"⚙️ [3/3] 프레임 추론 및 렌더링 시작 (총 처리 예정: {target_total} 프레임)...")
    start_time = time.time()
    frame_idx = 0
    processed_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # 프레임 건너뛰기 적용 시
            if frame_idx % frame_stride != 0:
                continue

            frame_start = time.time()

            # SAHI 추론 및 시각화 적용
            annotated_frame = annotate_frame_with_sahi(
                frame=frame,
                detection_model=detection_model,
                target_classes=target_classes,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
                frame_idx=frame_idx,
                total_frames=target_total
            )

            # 비디오 파일에 프레임 쓰기
            out.write(annotated_frame)
            processed_count += 1

            frame_elapsed = time.time() - frame_start
            total_elapsed = time.time() - start_time
            avg_fps = processed_count / total_elapsed if total_elapsed > 0 else 0
            percent = (frame_idx / target_total) * 100 if target_total > 0 else 0

            # 진행 상황 콘솔 출력 (한 줄 갱신 또는 줄바꿈)
            print(
                f"\r▶ 처리 중: {frame_idx}/{target_total} 프레임 ({percent:5.1f}%) | "
                f"속도: {avg_fps:4.1f} FPS | 소요: {total_elapsed:5.1f}초",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 작업이 중단되었습니다. 현재까지의 영상을 안전하게 저장합니다.")
    finally:
        cap.release()
        out.release()

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 [처리 완료] 총 {processed_count} 프레임 렌더링 완료!")
    print(f" • 총 소요 시간: {total_time:.2f}초 (평균 {processed_count / total_time:.2f} FPS)")
    print(f" • 저장된 결과 파일: {output_video_path}")
    print("=" * 70)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input_videos"
    output_dir = base_dir / "output_results"

    # 두 번째 영상 (후방 카메라 CCTV 시점)
    input_file = str(input_dir / "Hailuo_Video_Static CCTV camera view from t_568477421927383045.mp4")
    output_file = str(output_dir / "두번째영상_analyzed.mp4")

    # SAHI 동영상 검출 실행 (3프레임 간격 추론 최적화 적용)
    run_sahi_video_detection(
        input_video_path=input_file,
        output_video_path=output_file,
        model_path="yolov8x.pt",     # 높은 인식률 유지를 위해 x 모델 그대로 유지
        slice_size=640,
        overlap_ratio=0.2,
        conf_threshold=0.25,
        max_frames=None,
        frame_stride=3               # [핵심 최적화] 3프레임당 1번만 분석하여 연산 속도 3배 향상!
    )

