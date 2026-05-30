from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.models import game as game_models  # noqa: F401
from app.services.game_service import GameService


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
    assert sum(item["totalStages"] for item in games) == 30


def test_game_service_progresses_on_correct_answer() -> None:
    service = _make_service()

    state = service.start_game(1, "arithmetic")
    assert state["progress"]["currentStageNo"] == 1

    result = service.submit_answer(1, "arithmetic", 1, {"value": 2})

    assert result["correct"] is True
    assert result["progress"]["currentStageNo"] == 2
    assert result["progress"]["score"] == 100
    assert result["completed"] is False
