from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.models import game as game_models  # noqa: F401
from app.models.game import GameCatalog, GameStage
from app.services.game_service import (
    STAGES_PER_GAME,
    GameService,
    _from_json,
    _initials,
    _maze_reaches_goal,
    build_game_seed_data,
)


def _make_service() -> GameService:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return GameService(SessionLocal())


def test_game_service_seeds_catalog_and_stages() -> None:
    service = _make_service()

    games = service.list_games()

    assert len(games) == 4
    assert {item["slug"] for item in games} == {"memory_match", "maze", "arithmetic", "initials_quiz"}
    assert all(item["totalStages"] == STAGES_PER_GAME for item in games)


def test_initials_stages_have_exactly_one_matching_option() -> None:
    answers = set()
    for seed in build_game_seed_data():
        if seed.game_slug != "initials_quiz":
            continue
        clue = seed.payload["clue"]
        matching = [opt for opt in seed.payload["options"] if _initials(opt) == clue]
        assert matching == [seed.answer["value"]], seed
        assert len(set(seed.payload["options"])) == 4
        answers.add(seed.answer["value"])
    assert len(answers) == STAGES_PER_GAME


def test_maze_stages_are_solvable_with_expected_path() -> None:
    for seed in build_game_seed_data():
        if seed.game_slug != "maze":
            continue
        assert _maze_reaches_goal(seed.payload["grid"], seed.answer["path"]), seed.stage_no
        assert not _maze_reaches_goal(seed.payload["grid"], seed.answer["path"][:-1])


def test_arithmetic_options_contain_answer() -> None:
    for seed in build_game_seed_data():
        if seed.game_slug != "arithmetic":
            continue
        assert seed.answer["value"] in seed.payload["options"]
        assert len(set(seed.payload["options"])) == 4


def test_submit_advances_and_scores_only_when_correct() -> None:
    service = _make_service()
    state = service.start_game(1, "arithmetic")
    assert state["progress"]["currentStageNo"] == 1
    correct_value = _from_json(service._stage_query("arithmetic", 1).answer_json)["value"]

    wrong = service.submit_answer(1, "arithmetic", 1, {"value": str(correct_value + 1000)})
    assert wrong["correct"] is False
    assert wrong["progress"]["currentStageNo"] == 2
    assert wrong["progress"]["score"] == 0
    assert "정답은" in wrong["message"]

    correct_value = _from_json(service._stage_query("arithmetic", 2).answer_json)["value"]
    right = service.submit_answer(1, "arithmetic", 2, {"value": str(correct_value)})
    assert right["correct"] is True
    assert right["progress"]["currentStageNo"] == 3
    assert right["progress"]["score"] == 100
    assert "+100점" in right["message"]

    summary = service.list_progress_for_user(1)
    assert summary["totalScore"] == 100
    assert len(summary["games"]) == 4
    assert summary["games"][0]["progress"] is None or summary["games"][0]["game"]["slug"] == "memory_match"


def test_reseeds_when_stage_count_changes() -> None:
    service = _make_service()
    service.list_games()
    # 예전 시드(문제 수가 적은 상태)를 흉내 낸다.
    service.db.query(GameStage).filter(GameStage.stage_no > 8).delete()
    for catalog in service.db.query(GameCatalog).all():
        catalog.total_stages = 8
        service.db.add(catalog)
    service.db.commit()

    games = service.list_games()

    assert all(item["totalStages"] == STAGES_PER_GAME for item in games)
    assert service.db.query(GameStage).count() == STAGES_PER_GAME * 4


def test_reset_keeps_score_and_activity_is_grouped_by_day() -> None:
    service = _make_service()
    service.start_game(1, "arithmetic")
    correct_value = _from_json(service._stage_query("arithmetic", 1).answer_json)["value"]
    service.submit_answer(1, "arithmetic", 1, {"value": str(correct_value)})
    service.submit_answer(1, "arithmetic", 2, {"value": "-1"})

    reset = service.reset_game(1, "arithmetic")
    assert reset["progress"]["currentStageNo"] == 1
    assert reset["progress"]["score"] == 100
    assert reset["progress"]["attempts"] == 2

    activity = service.list_activity_for_user(1, 30)
    assert len(activity["activity"]) == 1
    day = activity["activity"][0]
    assert day["attempts"] == 2 and day["correct"] == 1 and day["score"] == 100
