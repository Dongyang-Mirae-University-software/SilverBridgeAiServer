"""대화 자동 정리: 마지막 활동 뒤 제한 시간이 지난 대화만 메모리에서 지운다."""

from app.services.companion.dialogue_store import IDLE_LIMIT_SEC, DialogueStore


def test_default_idle_limit_matches_be_ten_minutes():
    assert IDLE_LIMIT_SEC == 10 * 60  # BE 자동 종료 기준(10분)과 맞춘다


def _make(store: DialogueStore, ward: str = "w"):
    return store.create(ward, "s", "09:00", "잘 주무셨어요?", 8)


def test_idle_dialogue_is_removed_when_a_new_one_is_created():
    store = DialogueStore(idle_limit_sec=60)
    old = _make(store)
    old.last_active -= 120  # 2분 동안 아무 활동이 없었다고 본다
    new = _make(store)
    assert store.get(old.dialogueId) is None
    assert store.get(new.dialogueId) is not None


def test_recently_used_dialogue_is_kept():
    store = DialogueStore(idle_limit_sec=60)
    d = _make(store)
    d.last_active -= 50  # 아직 제한 안이다
    store.add_turn(d.dialogueId, "기쁨", "TEXT", no_response_limit=3)  # 활동이 생겨 시간이 갱신된다
    _make(store)
    assert store.get(d.dialogueId) is not None


def test_finished_but_idle_dialogue_is_also_removed():
    store = DialogueStore(idle_limit_sec=60)
    d = _make(store)
    store.finish(d.dialogueId, "USER_STOP")
    d.last_active -= 120
    _make(store)
    assert store.get(d.dialogueId) is None
