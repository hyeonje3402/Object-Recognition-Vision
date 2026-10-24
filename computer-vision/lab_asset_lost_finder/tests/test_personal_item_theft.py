"""
개인 소지품 절도 판정 엔진 시나리오 테스트 (합성 데이터, 영상/모델 불필요)

실행: python tests/test_personal_item_theft.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.personal_item_tracker import PersonalItemTheftDetector, PersonalState

FPS = 10.0
ITEM = {"ITEM_01": {"class": "laptop", "bbox": [485, 490, 515, 510]}}   # 중심 (500, 500)
DOOR = (900, 0, 1000, 600)
OWNER_NEAR = (560, 520)   # 물품과 거리 30px
FAR = (100, 520)          # 물품과 거리 > 150px
THIEF_NEAR = (440, 520)


def box(x, y):
    return [x - 30, y - 200, x + 30, y]


def lerp(a, b, n):
    return [(a[0] + (b[0] - a[0]) * i / max(1, n - 1), a[1] + (b[1] - a[1]) * i / max(1, n - 1))
            for i in range(n)]


class Sim:
    """프레임 시뮬레이터: seg()로 구간을 이어 붙인다."""

    def __init__(self, **kw):
        self.det = PersonalItemTheftDetector(exit_door_bbox=DOOR, fps=FPS, **kw)
        self.frame = 0
        self.events = []

    def seg(self, n, present, people):
        """people: {id: (x,y) 고정 위치 | [(x,y)...] 길이 n의 경로}"""
        for i in range(n):
            self.frame += 1
            persons = []
            for pid, pos in people.items():
                x, y = pos[i] if isinstance(pos, list) else pos
                persons.append({"track_id": pid, "bbox": box(x, y)})
            self.events += self.det.update(self.frame, ITEM, {"ITEM_01": present}, persons)

    def types(self):
        return [e["event_type"] for e in self.events]


def owner_sits(sim, n=20):
    sim.seg(n, True, {1: OWNER_NEAR})


def test_owner_takes_own_item():
    s = Sim()
    owner_sits(s)
    s.seg(4, False, {1: OWNER_NEAR})
    s.seg(40, False, {1: lerp(OWNER_NEAR, (950, 300), 40)})
    assert s.types() == [], s.types()


def test_third_party_theft_detected():
    s = Sim()
    owner_sits(s)
    s.seg(25, True, {1: lerp(OWNER_NEAR, FAR, 25)})
    s.seg(15, True, {1: FAR, 2: THIEF_NEAR})
    s.seg(4, False, {1: FAR, 2: THIEF_NEAR})
    s.seg(40, False, {1: FAR, 2: lerp(THIEF_NEAR, (950, 300), 40)})
    assert s.types() == ["PERSONAL_THEFT"], s.types()
    ev = s.events[0]
    assert ev["owner_id"] == 1 and ev["suspect_id"] == 2, ev


def test_passerby_not_suspect():
    s = Sim()
    owner_sits(s)
    s.seg(25, True, {1: lerp(OWNER_NEAR, FAR, 25)})
    s.seg(16, True, {1: FAR, 3: lerp((100, 700), (900, 700), 16)})  # 빠르게 스쳐 지나감
    s.seg(30, True, {1: FAR})
    assert s.types() == [], s.types()
    assert s.det.get_state("ITEM_01") == PersonalState.OWNER_AWAY


def test_item_returned_no_alert():
    s = Sim()
    owner_sits(s)
    s.seg(25, True, {1: lerp(OWNER_NEAR, FAR, 25)})
    s.seg(15, True, {1: FAR, 2: THIEF_NEAR})
    s.seg(20, False, {1: FAR, 2: THIEF_NEAR})            # 잠시 집었다가
    s.seg(10, True, {1: FAR, 2: THIEF_NEAR})             # 다시 내려놓음
    s.seg(40, True, {1: FAR, 2: lerp(THIEF_NEAR, (950, 300), 40)})
    assert s.types() == [], s.types()


def test_owner_id_switch_not_theft():
    s = Sim()
    owner_sits(s)                                        # ID 1 이 주인
    s.seg(5, True, {})                                   # 트래킹 끊김
    s.seg(12, True, {9: OWNER_NEAR})                     # 같은 자리에 새 ID 9 (ID 스위칭)
    s.seg(4, False, {9: OWNER_NEAR})
    s.seg(40, False, {9: lerp(OWNER_NEAR, (950, 300), 40)})
    assert s.types() == [], s.types()
    assert s.det.get_owner("ITEM_01") == 9


def test_vanish_without_anyone_no_alert():
    s = Sim()
    owner_sits(s)
    s.seg(25, True, {1: lerp(OWNER_NEAR, FAR, 25)})
    s.seg(40, False, {1: FAR})                           # 조명 노이즈 등으로 소멸
    assert s.types() == [], s.types()


def test_owner_present_ambiguity_defers():
    s = Sim()
    owner_sits(s)
    s.seg(15, True, {1: OWNER_NEAR, 2: THIEF_NEAR})
    s.seg(4, False, {1: OWNER_NEAR, 2: THIEF_NEAR})
    s.seg(40, False, {1: OWNER_NEAR, 2: lerp(THIEF_NEAR, (950, 300), 40)})
    assert s.types() == [], s.types()


def test_not_heading_to_door_only_unresolved():
    s = Sim()
    owner_sits(s)
    s.seg(25, True, {1: lerp(OWNER_NEAR, FAR, 25)})
    s.seg(15, True, {1: FAR, 2: THIEF_NEAR})
    s.seg(4, False, {1: FAR, 2: THIEF_NEAR})
    s.seg(320, False, {1: FAR, 2: lerp(THIEF_NEAR, (100, 700), 320)})  # 출입문 반대 방향
    assert s.types() == ["PERSONAL_MISSING"], s.types()


def test_abandoned_then_stolen():
    s = Sim()
    owner_sits(s)
    s.seg(25, True, {1: lerp(OWNER_NEAR, FAR, 25)})
    s.seg(120, True, {})                                 # 주인 퇴실, 방치
    s.seg(15, True, {2: THIEF_NEAR})
    s.seg(4, False, {2: THIEF_NEAR})
    s.seg(40, False, {2: lerp(THIEF_NEAR, (950, 300), 40)})
    assert s.types() == ["PERSONAL_ABANDONED", "PERSONAL_THEFT"], s.types()


def test_unowned_item_never_alerts():
    s = Sim()
    s.seg(10, True, {})                                  # 주인이 확정되기 전
    s.seg(15, True, {2: THIEF_NEAR})
    s.seg(4, False, {2: THIEF_NEAR})
    s.seg(40, False, {2: lerp(THIEF_NEAR, (950, 300), 40)})
    # 2번이 자리에 머문 시간이 own_sec 이상이면 2번이 주인이 되므로 절도가 아님
    assert s.types() == [], s.types()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
