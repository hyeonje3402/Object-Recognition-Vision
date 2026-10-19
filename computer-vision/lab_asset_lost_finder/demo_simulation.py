"""
[가상 시뮬레이션 테스트 - demo_simulation.py]
웹캠 없이도 룰 1(기자재 반출) 및 룰 2(개인 분실물 방치) 상태 머신이
정상 작동하는지 합성 프레임으로 검증하는 스크립트
"""
import cv2
import numpy as np
import time
from pathlib import Path
from main import LabSecuritySystem
from core.state_machine import AssetState, LostState

def run_simulation():
    print("=" * 65)
    print("🧪 [테스트 시뮬레이션] 실습실 보안 & 분실물 상태 머신 검증")
    print("=" * 65)

    system = LabSecuritySystem(source=0)
    w, h = 1280, 720
    rois = system.get_pixel_rois(w, h)

    # 4단계 시나리오 시뮬레이션
    scenarios = [
        {"desc": "1단계: 정상 상태 (교탁에 마우스 존재, 학생이 책상에서 물품 소지)",
         "mouse_on_podium": True, "person_at_desk": True, "person_at_door": False, "duration": 2.0},
        {"desc": "2단계: 주의 상태 (교탁 마우스 분실 감지 -> 타이머 카운트다운 시작)",
         "mouse_on_podium": False, "person_at_desk": True, "person_at_door": False, "duration": 3.0},
        {"desc": "3단계: 반출 경고 (마우스 없는 상태에서 출입문 ROI에 사람 감지 -> THEFT ALERT)",
         "mouse_on_podium": False, "person_at_desk": True, "person_at_door": True, "duration": 3.0},
        {"desc": "4단계: 분실물 방치 (학생이 책상에서 이탈 -> 개인물품 LOST ALERT 확정)",
         "mouse_on_podium": True, "person_at_desk": False, "person_at_door": False, "duration": 9.0},
    ]

    output_dir = Path(__file__).resolve().parent / "demo_output"
    output_dir.mkdir(exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_dir / "state_machine_demo.mp4"), fourcc, 10, (w, h))

    total_steps = 0
    for sc in scenarios:
        print(f"\n▶ 진행 중: {sc['desc']}")
        start = time.time()
        while time.time() - start < sc["duration"]:
            total_steps += 1
            # 빈 실습실 배경 생성
            frame = np.full((h, w, 3), 40, dtype=np.uint8)

            # 교탁 및 책상 영역 기본 가이드
            cv2.rectangle(frame, (rois["PODIUM"][0], rois["PODIUM"][1]), (rois["PODIUM"][2], rois["PODIUM"][3]), (70, 70, 70), -1)
            cv2.rectangle(frame, (rois["STUDENT_DESK"][0], rois["STUDENT_DESK"][1]), (rois["STUDENT_DESK"][2], rois["STUDENT_DESK"][3]), (60, 60, 60), -1)
            cv2.rectangle(frame, (rois["DOOR"][0], rois["DOOR"][1]), (rois["DOOR"][2], rois["DOOR"][3]), (50, 50, 80), -1)

            # 상태 머신 업데이트
            mouse_state = system.mouse_asset_sm.update(
                is_detected_in_podium=sc["mouse_on_podium"],
                is_person_at_door=sc["person_at_door"]
            )

            # 책상 위 개인물품 상태 머신
            item_id = 1
            if item_id not in system.lost_items_sm:
                system.lost_items_sm[item_id] = system.lost_items_sm.get(item_id, None) or \
                    type(system.mouse_asset_sm)(asset_name="Tint", timeout_sec=8.0) # LostPropertyStateMachine 대체
            
            # 개인물품 상태 머신 업데이트
            from core.state_machine import LostPropertyStateMachine
            if not isinstance(system.lost_items_sm.get(item_id), LostPropertyStateMachine):
                system.lost_items_sm[item_id] = LostPropertyStateMachine(item_id=1, item_class="tint", timeout_sec=8.0)
            lost_state = system.lost_items_sm[item_id].update(is_person_nearby=sc["person_at_desk"])

            # 시각화 박스 추가
            if sc["mouse_on_podium"]:
                cv2.rectangle(frame, (rois["PODIUM"][0]+50, rois["PODIUM"][1]+50),
                              (rois["PODIUM"][0]+120, rois["PODIUM"][1]+100), (0, 255, 0), -1)
                cv2.putText(frame, "Public Mouse", (rois["PODIUM"][0]+50, rois["PODIUM"][1]+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 개인물품 틴트/아이패드
            item_col = (0, 255, 0) if lost_state == LostState.ATTENDED else ((0, 200, 255) if lost_state == LostState.UNATTENDED else (0, 0, 255))
            cv2.rectangle(frame, (rois["STUDENT_DESK"][0]+100, rois["STUDENT_DESK"][1]+80),
                          (rois["STUDENT_DESK"][0]+180, rois["STUDENT_DESK"][1]+140), item_col, 2)
            cv2.putText(frame, f"Small Item: Tint [{lost_state.value}]",
                        (rois["STUDENT_DESK"][0]+70, rois["STUDENT_DESK"][1]+70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, item_col, 2)

            if sc["person_at_desk"]:
                cv2.rectangle(frame, (rois["STUDENT_DESK"][0]+220, rois["STUDENT_DESK"][1]+30),
                              (rois["STUDENT_DESK"][0]+380, rois["STUDENT_DESK"][1]+250), (255, 180, 0), 2)
                cv2.putText(frame, "Student (Person)", (rois["STUDENT_DESK"][0]+220, rois["STUDENT_DESK"][1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

            if sc["person_at_door"]:
                cv2.rectangle(frame, (rois["DOOR"][0]+20, rois["DOOR"][1]+50),
                              (rois["DOOR"][2]-20, rois["DOOR"][3]-50), (0, 0, 255), 2)
                cv2.putText(frame, "Person at Door!", (rois["DOOR"][0]+10, rois["DOOR"][1]+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # HUD 렌더링
            system.draw_hud(frame, rois)
            out_video.write(frame)
            time.sleep(0.08)

    out_video.release()
    print("\n✅ 시뮬레이션 완료! 결과 비디오 저장:")
    print(output_dir / "state_machine_demo.mp4")

if __name__ == "__main__":
    run_simulation()
