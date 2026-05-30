from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from html import escape
from threading import Lock
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.core.response import error_response
from app.models.game import GameAttempt, GameCatalog, GameProgress, GameStage


SEED_LOCK = Lock()


@dataclass(frozen=True)
class StageSeed:
    game_slug: str
    stage_no: int
    title: str
    stage_type: str
    prompt: str
    payload: dict[str, Any]
    answer: dict[str, Any]
    max_score: int = 100


def _now() -> datetime:
    return datetime.utcnow()


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _from_json(raw: str | None, default: Any = None) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _normalize_answer_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_dir_path(value: Any) -> str:
    text = _normalize_answer_value(value).upper()
    return "".join(ch for ch in text if ch in {"U", "D", "L", "R"})


def _normalize_pair_keys(answer: Any) -> set[str]:
    if isinstance(answer, dict):
        candidates = answer.get("matchedPairs") or answer.get("pairs") or answer.get("answers") or []
    else:
        candidates = answer or []
    if isinstance(candidates, str):
        candidates = [candidates]
    return {str(item).strip() for item in candidates if str(item).strip()}


def _normalize_option(answer: Any) -> str:
    if isinstance(answer, dict):
        if "value" in answer:
            return _normalize_answer_value(answer["value"])
        if "answer" in answer:
            return _normalize_answer_value(answer["answer"])
        if "selectedIndex" in answer:
            return _normalize_answer_value(answer["selectedIndex"])
        if "choiceIndex" in answer:
            return _normalize_answer_value(answer["choiceIndex"])
    return _normalize_answer_value(answer)


def _maze_grid_from_path(path: str, size: int = 6) -> tuple[list[list[str]], tuple[int, int], tuple[int, int], str]:
    moves = _normalize_dir_path(path)
    x = 0
    y = 0
    coords = [(x, y)]
    for move in moves:
        if move == "R":
            x += 1
        elif move == "L":
            x -= 1
        elif move == "D":
            y += 1
        elif move == "U":
            y -= 1
        if x < 0 or y < 0 or x >= size or y >= size:
            raise ValueError(f"Invalid maze path: {path}")
        coords.append((x, y))

    grid = [["#" for _ in range(size)] for _ in range(size)]
    for cx, cy in coords:
        grid[cy][cx] = "."
    start = coords[0]
    goal = coords[-1]
    grid[start[1]][start[0]] = "S"
    grid[goal[1]][goal[0]] = "G"
    return grid, start, goal, moves


def _memory_stage(stage_no: int, labels: list[str]) -> StageSeed:
    cards: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        cards.append({"id": f"m{stage_no}-{idx}-a", "pairKey": label, "label": label})
        cards.append({"id": f"m{stage_no}-{idx}-b", "pairKey": label, "label": label})
    random.Random(9000 + stage_no).shuffle(cards)
    return StageSeed(
        game_slug="memory_match",
        stage_no=stage_no,
        title=f"짝맞추기 {stage_no}",
        stage_type="memory_match",
        prompt=f"{len(labels)}쌍의 카드를 모두 맞추세요.",
        payload={
            "kind": "memory_match",
            "cards": cards,
            "pairCount": len(labels),
            "columns": 4 if len(cards) <= 8 else 6,
        },
        answer={"requiredPairs": labels},
    )


def _maze_stage(stage_no: int, path: str) -> StageSeed:
    grid, start, goal, moves = _maze_grid_from_path(path, size=6)
    return StageSeed(
        game_slug="maze",
        stage_no=stage_no,
        title=f"미로찾기 {stage_no}",
        stage_type="maze",
        prompt="화살표 버튼으로 출구까지 이동하세요.",
        payload={
            "kind": "maze",
            "grid": grid,
            "start": {"x": start[0], "y": start[1]},
            "goal": {"x": goal[0], "y": goal[1]},
            "solutionLength": len(moves),
        },
        answer={"path": moves},
    )


def _arithmetic_stage(stage_no: int, left: int, op: str, right: int, options: list[int]) -> StageSeed:
    if op == "+":
        correct = left + right
    elif op == "-":
        correct = left - right
    elif op == "×":
        correct = left * right
    elif op == "÷":
        correct = left // right
    else:
        raise ValueError(f"Unsupported operator: {op}")
    return StageSeed(
        game_slug="arithmetic",
        stage_no=stage_no,
        title=f"사칙연산 {stage_no}",
        stage_type="arithmetic",
        prompt="정답을 선택하세요.",
        payload={
            "kind": "arithmetic",
            "question": f"{left} {op} {right} = ?",
            "options": options,
        },
        answer={"value": correct},
    )


def _initials_stage(stage_no: int, clue: str, answer_word: str, options: list[str]) -> StageSeed:
    return StageSeed(
        game_slug="initials_quiz",
        stage_no=stage_no,
        title=f"초성퀴즈 {stage_no}",
        stage_type="initials_quiz",
        prompt="초성과 가장 잘 맞는 단어를 고르세요.",
        payload={
            "kind": "initials_quiz",
            "clue": clue,
            "options": options,
        },
        answer={"value": answer_word},
    )


def build_game_seed_data() -> list[StageSeed]:
    seeds: list[StageSeed] = []
    seeds.extend(
        [
            _memory_stage(1, ["사과", "별", "고양이"]),
            _memory_stage(2, ["바다", "하늘", "꽃"]),
            _memory_stage(3, ["달", "해", "나무", "물"]),
            _memory_stage(4, ["기차", "버스", "자동차", "자전거"]),
            _memory_stage(5, ["연필", "지우개", "공책", "가방", "책"]),
            _memory_stage(6, ["의자", "책상", "램프", "창문", "문"]),
            _memory_stage(7, ["토끼", "사자", "호랑이", "기린", "코끼리"]),
            _memory_stage(8, ["서울", "부산", "대구", "광주", "대전", "울산"]),
        ],
    )
    seeds.extend(
        [
            _maze_stage(1, "RRDD"),
            _maze_stage(2, "RDDR"),
            _maze_stage(3, "RRRDDD"),
            _maze_stage(4, "RDRRDD"),
            _maze_stage(5, "RRDDRR"),
            _maze_stage(6, "RRRDRD"),
            _maze_stage(7, "RDDRDR"),
            _maze_stage(8, "RRDRDD"),
        ],
    )
    seeds.extend(
        [
            _arithmetic_stage(1, 7, "+", 5, [10, 11, 12, 13]),
            _arithmetic_stage(2, 9, "-", 3, [4, 6, 8, 9]),
            _arithmetic_stage(3, 4, "×", 6, [20, 22, 24, 26]),
            _arithmetic_stage(4, 18, "÷", 3, [5, 6, 7, 9]),
            _arithmetic_stage(5, 8, "+", 7, [9, 11, 12, 13]),
            _arithmetic_stage(6, 3, "×", 3, [8, 10, 11, 12]),
            _arithmetic_stage(7, 20, "-", 10, [6, 8, 10, 12]),
        ],
    )
    seeds.extend(
        [
            _initials_stage(1, "ㄱㅂ", "가방", ["가방", "기분", "고기", "기차"]),
            _initials_stage(2, "ㅊㅂ", "초밥", ["초밥", "책상", "출발", "차별"]),
            _initials_stage(3, "ㅂㄷ", "바다", ["바다", "버스", "방울", "배달"]),
            _initials_stage(4, "ㅅㄱ", "사과", ["사과", "시계", "소금", "선글라스"]),
            _initials_stage(5, "ㅇㅇ", "우유", ["우유", "의자", "외출", "오이"]),
            _initials_stage(6, "ㅎㄱ", "학교", ["학교", "하구", "햇갈", "호기"]),
            _initials_stage(7, "ㅇㅍ", "연필", ["연필", "염필", "영팔", "연표"]),
        ],
    )
    return seeds


class GameService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _ensure_seeded(self) -> None:
        with SEED_LOCK:
            if self.db.query(GameCatalog).count() > 0:
                return
            catalog_map: dict[str, dict[str, Any]] = {
                "memory_match": {
                    "title": "짝맞추기",
                    "description": "같은 그림 카드를 짝으로 맞추는 게임입니다.",
                    "theme_color": "#7c3aed",
                },
                "maze": {
                    "title": "미로찾기",
                    "description": "출구까지 최단 경로로 이동하는 게임입니다.",
                    "theme_color": "#0ea5e9",
                },
                "arithmetic": {
                    "title": "사칙연산",
                    "description": "덧셈, 뺄셈, 곱셈, 나눗셈 문제를 풀어보세요.",
                    "theme_color": "#f97316",
                },
                "initials_quiz": {
                    "title": "초성퀴즈",
                    "description": "초성을 보고 정답 단어를 맞히는 게임입니다.",
                    "theme_color": "#22c55e",
                },
            }
            catalogs = {
                slug: GameCatalog(
                    slug=slug,
                    title=spec["title"],
                    description=spec["description"],
                    total_stages=0,
                    theme_color=spec["theme_color"],
                )
                for slug, spec in catalog_map.items()
            }
            stages = build_game_seed_data()
            for seed in stages:
                catalogs[seed.game_slug].total_stages += 1

            for catalog in catalogs.values():
                self.db.add(catalog)
            self.db.flush()

            for seed in stages:
                self.db.add(
                    GameStage(
                        game_slug=seed.game_slug,
                        stage_no=seed.stage_no,
                        title=seed.title,
                        stage_type=seed.stage_type,
                        prompt=seed.prompt,
                        payload_json=_to_json(seed.payload),
                        answer_json=_to_json(seed.answer),
                        max_score=seed.max_score,
                    ),
                )
            self.db.commit()

    def _catalog_query(self, game_slug: str) -> GameCatalog:
        self._ensure_seeded()
        catalog = self.db.query(GameCatalog).filter(GameCatalog.slug == game_slug).first()
        if not catalog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_response("게임을 찾을 수 없습니다.", "GAME_NOT_FOUND", None),
            )
        return catalog

    def _stage_query(self, game_slug: str, stage_no: int) -> GameStage:
        stage = (
            self.db.query(GameStage)
            .filter(GameStage.game_slug == game_slug, GameStage.stage_no == stage_no)
            .first()
        )
        if not stage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_response("스테이지를 찾을 수 없습니다.", "GAME_STAGE_NOT_FOUND", None),
            )
        return stage

    def _progress_query(self, user_id: int, game_slug: str) -> GameProgress | None:
        return (
            self.db.query(GameProgress)
            .filter(GameProgress.user_id == user_id, GameProgress.game_slug == game_slug)
            .first()
        )

    def _create_progress(self, user_id: int, game_slug: str) -> GameProgress:
        progress = GameProgress(
            user_id=user_id,
            game_slug=game_slug,
            current_stage_no=1,
            score=0,
            attempts=0,
            cleared=False,
            last_answer_correct=None,
            state_json=_to_json({"currentStageNo": 1, "score": 0, "cleared": False}),
            started_at=_now(),
            updated_at=_now(),
        )
        self.db.add(progress)
        self.db.commit()
        self.db.refresh(progress)
        return progress

    def _sync_progress_state(self, progress: GameProgress) -> None:
        progress.updated_at = _now()
        self.db.add(progress)
        self.db.commit()
        self.db.refresh(progress)

    @staticmethod
    def _catalog_payload(catalog: GameCatalog) -> dict[str, Any]:
        return {
            "slug": catalog.slug,
            "title": catalog.title,
            "description": catalog.description,
            "totalStages": catalog.total_stages,
            "themeColor": catalog.theme_color,
        }

    @staticmethod
    def _stage_payload(stage: GameStage) -> dict[str, Any]:
        return {
            "stageNo": stage.stage_no,
            "title": stage.title,
            "stageType": stage.stage_type,
            "prompt": stage.prompt,
            "payload": _from_json(stage.payload_json, {}),
            "maxScore": stage.max_score,
        }

    @staticmethod
    def _progress_payload(progress: GameProgress) -> dict[str, Any]:
        return {
            "userId": progress.user_id,
            "gameSlug": progress.game_slug,
            "currentStageNo": progress.current_stage_no,
            "score": progress.score,
            "attempts": progress.attempts,
            "cleared": progress.cleared,
            "lastAnswerCorrect": progress.last_answer_correct,
            "state": _from_json(progress.state_json, {}),
            "startedAt": _iso(progress.started_at),
            "updatedAt": _iso(progress.updated_at),
            "clearedAt": _iso(progress.cleared_at),
        }

    def list_games(self) -> list[dict[str, Any]]:
        self._ensure_seeded()
        rows = self.db.query(GameCatalog).order_by(GameCatalog.id.asc()).all()
        return [self._catalog_payload(row) for row in rows]

    def start_game(self, user_id: int, game_slug: str) -> dict[str, Any]:
        catalog = self._catalog_query(game_slug)
        progress = self._progress_query(user_id, game_slug)
        if progress is None:
            progress = self._create_progress(user_id, game_slug)
        state = self.get_state_payload(user_id, game_slug, progress=progress, catalog=catalog)
        return {
            "game": self._catalog_payload(catalog),
            "progress": state["progress"],
            "stage": state["stage"],
            "totalStages": catalog.total_stages,
            "completed": progress.cleared,
            "iframeUrl": f"/api/v1/games/embed?userId={user_id}&gameSlug={game_slug}",
        }

    def get_state_payload(
        self,
        user_id: int,
        game_slug: str,
        *,
        progress: GameProgress | None = None,
        catalog: GameCatalog | None = None,
    ) -> dict[str, Any]:
        catalog = catalog or self._catalog_query(game_slug)
        progress = progress or self._progress_query(user_id, game_slug) or self._create_progress(user_id, game_slug)
        stage = None
        if not progress.cleared and progress.current_stage_no <= catalog.total_stages:
            current_stage = self._stage_query(game_slug, progress.current_stage_no)
            stage = self._stage_payload(current_stage)
        return {
            "game": self._catalog_payload(catalog),
            "progress": self._progress_payload(progress),
            "stage": stage,
            "totalStages": catalog.total_stages,
            "completed": progress.cleared,
            "iframeUrl": f"/api/v1/games/embed?userId={user_id}&gameSlug={game_slug}",
        }

    def _evaluate_stage(self, stage: GameStage, answer: Any) -> tuple[bool, int, str]:
        expected = _from_json(stage.answer_json, {})
        stage_type = stage.stage_type

        if stage_type == "memory_match":
            submitted_pairs = _normalize_pair_keys(answer)
            required_pairs = {str(item).strip() for item in expected.get("requiredPairs", []) if str(item).strip()}
            correct = bool(required_pairs) and submitted_pairs == required_pairs
            message = "모든 짝을 맞췄습니다." if correct else "맞춘 짝을 다시 확인해 주세요."
        elif stage_type == "maze":
            submitted_path = _normalize_dir_path(answer.get("path") if isinstance(answer, dict) else answer)
            expected_path = _normalize_dir_path(expected.get("path"))
            correct = submitted_path == expected_path
            message = "출구에 도착했습니다." if correct else "미로 경로가 다릅니다."
        elif stage_type == "arithmetic":
            expected_value = _normalize_answer_value(expected.get("value"))
            submitted_value = _normalize_answer_value(answer.get("value") if isinstance(answer, dict) else answer)
            payload = _from_json(stage.payload_json, {})
            options = payload.get("options") or []
            if submitted_value.isdigit():
                option_index = int(submitted_value)
                if 0 <= option_index < len(options):
                    correct = _normalize_answer_value(options[option_index]) == expected_value
                else:
                    correct = submitted_value == expected_value
            else:
                correct = submitted_value == expected_value
            message = "정답입니다." if correct else "다시 계산해 보세요."
        elif stage_type == "initials_quiz":
            submitted_value = _normalize_option(answer)
            expected_value = _normalize_answer_value(expected.get("value"))
            if submitted_value.isdigit():
                option_index = int(submitted_value)
                payload = _from_json(stage.payload_json, {})
                options = payload.get("options") or []
                if 0 <= option_index < len(options):
                    correct = _normalize_answer_value(options[option_index]) == expected_value
                else:
                    correct = submitted_value == expected_value
            else:
                correct = submitted_value == expected_value
            message = "정답입니다." if correct else "다시 고르세요."
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_response("지원하지 않는 게임 타입입니다.", "GAME_STAGE_INVALID_TYPE", None),
            )

        score_delta = stage.max_score if correct else 0
        return correct, score_delta, message

    def submit_answer(self, user_id: int, game_slug: str, stage_no: int | None, answer: Any) -> dict[str, Any]:
        catalog = self._catalog_query(game_slug)
        progress = self._progress_query(user_id, game_slug)
        if progress is None:
            progress = self._create_progress(user_id, game_slug)

        target_stage_no = stage_no or progress.current_stage_no
        if progress.cleared:
            return {
                "correct": True,
                "message": "이미 모든 스테이지를 완료했습니다.",
                "scoreDelta": 0,
                "progress": self._progress_payload(progress),
                "stage": None,
                "completed": True,
            }
        if target_stage_no != progress.current_stage_no:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_response("현재 진행 중인 스테이지가 아닙니다.", "GAME_STAGE_MISMATCH", None),
            )
        if target_stage_no > catalog.total_stages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_response("이미 완료된 게임입니다.", "GAME_ALREADY_COMPLETED", None),
            )

        stage = self._stage_query(game_slug, target_stage_no)
        correct, score_delta, message = self._evaluate_stage(stage, answer)
        progress.attempts += 1
        progress.last_answer_correct = correct

        if correct:
            progress.score += score_delta
            progress.current_stage_no += 1
            if progress.current_stage_no > catalog.total_stages:
                progress.cleared = True
                progress.cleared_at = _now()

        progress.state_json = _to_json(
            {
                "currentStageNo": progress.current_stage_no,
                "score": progress.score,
                "attempts": progress.attempts,
                "cleared": progress.cleared,
                "lastAnswerCorrect": progress.last_answer_correct,
                "lastStageNo": target_stage_no,
                "lastMessage": message,
            },
        )
        self._sync_progress_state(progress)

        attempt = GameAttempt(
            user_id=user_id,
            game_slug=game_slug,
            stage_no=target_stage_no,
            attempt_no=progress.attempts,
            answer_json=_to_json(answer),
            is_correct=correct,
            score_delta=score_delta,
            created_at=_now(),
        )
        self.db.add(attempt)
        self.db.commit()

        next_stage = None
        if not progress.cleared and progress.current_stage_no <= catalog.total_stages:
            next_stage_row = self._stage_query(game_slug, progress.current_stage_no)
            next_stage = self._stage_payload(next_stage_row)

        return {
            "correct": correct,
            "message": message,
            "scoreDelta": score_delta,
            "progress": self._progress_payload(progress),
            "stage": next_stage,
            "completed": progress.cleared,
        }

    def reset_game(self, user_id: int, game_slug: str) -> dict[str, Any]:
        catalog = self._catalog_query(game_slug)
        progress = self._progress_query(user_id, game_slug)
        if progress is None:
            progress = self._create_progress(user_id, game_slug)
        progress.current_stage_no = 1
        progress.score = 0
        progress.attempts = 0
        progress.cleared = False
        progress.last_answer_correct = None
        progress.cleared_at = None
        progress.state_json = _to_json({"currentStageNo": 1, "score": 0, "attempts": 0, "cleared": False})
        self._sync_progress_state(progress)
        stage = self._stage_payload(self._stage_query(game_slug, 1))
        return {
            "game": self._catalog_payload(catalog),
            "progress": self._progress_payload(progress),
            "stage": stage,
            "totalStages": catalog.total_stages,
            "completed": False,
            "iframeUrl": f"/api/v1/games/embed?userId={user_id}&gameSlug={game_slug}",
        }

    def list_progress_for_user(self, user_id: int) -> list[dict[str, Any]]:
        self._ensure_seeded()
        rows = (
            self.db.query(GameProgress)
            .filter(GameProgress.user_id == user_id)
            .order_by(GameProgress.updated_at.desc())
            .all()
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            catalog = self.db.query(GameCatalog).filter(GameCatalog.slug == row.game_slug).first()
            results.append(
                {
                    "game": self._catalog_payload(catalog) if catalog else {"slug": row.game_slug},
                    "progress": self._progress_payload(row),
                },
            )
        return results

    def render_embed_html(self, request: Request, user_id: int, game_slug: str) -> HTMLResponse:
        catalog = self._catalog_query(game_slug)
        forwarded_proto = str(request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
        forwarded_host = str(request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
        host = forwarded_host or str(request.headers.get("host") or request.url.hostname or "").strip()
        scheme = "https" if forwarded_proto == "https" else request.url.scheme
        api_origin = f"{scheme}://{host}".rstrip("/")
        boot = {
            "userId": user_id,
            "gameSlug": game_slug,
            "apiOrigin": api_origin,
            "game": self._catalog_payload(catalog),
        }
        template = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #fffaf0;
      --card-bg: #ffffff;
      --text: #1f2937;
      --muted: #4b5563;
      --primary: __PRIMARY__;
      --primary-hover: #ea580c;
      --success: #16a34a;
      --danger: #dc2626;
      --info: #2563eb;
      --border: #fed7aa;
      --radius-lg: 32px;
      --radius-md: 22px;
      --shadow: 0 10px 25px rgba(124, 45, 18, 0.1);
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 22px;
      line-height: 1.5;
    }
    .shell {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 24px;
      gap: 24px;
    }
    header.hero {
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 24px;
      background: var(--card-bg);
      border: 3px solid var(--border);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }
    .hero-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .hero-title {
      margin: 0;
      font-size: 34px;
      font-weight: 900;
      color: var(--primary);
    }
    .hero-info {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    .pill {
      background: #fff7ed;
      border: 1px solid var(--border);
      padding: 8px 18px;
      border-radius: 999px;
      font-size: 18px;
      font-weight: 700;
      color: #9a3412;
    }
    .progress-container {
      width: 100%;
      height: 24px;
      background: #fed7aa;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }
    .progress-bar {
      height: 100%;
      background: var(--primary);
      border-radius: 999px;
      transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .layout {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .game-card {
      background: var(--card-bg);
      border: 3px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 32px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 24px;
      flex: 1;
    }
    .stage-header { text-align: center; }
    .stage-title { font-size: 28px; font-weight: 800; margin: 0 0 12px; }
    .stage-prompt { font-size: 24px; color: var(--muted); margin: 0; font-weight: 500; }
    
    .question-box {
      font-size: 42px;
      font-weight: 900;
      padding: 40px;
      text-align: center;
      background: #f8fafc;
      border-radius: var(--radius-md);
      border: 2px dashed var(--border);
      margin: 12px 0;
    }
    
    .feedback {
      font-size: 26px;
      font-weight: 800;
      padding: 24px;
      border-radius: var(--radius-md);
      text-align: center;
      display: none;
    }
    .feedback.visible { display: block; }
    .feedback.ok { background: #f0fdf4; color: var(--success); border: 2px solid #bcf0da; }
    .feedback.bad { background: #fef2f2; color: var(--danger); border: 2px solid #fecaca; }
    .feedback.info { background: #eff6ff; color: var(--info); border: 2px solid #bfdbfe; }

    .board { width: 100%; }
    
    /* Memory Match Styles */
    .memory-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-top: 20px;
    }
    .memory-card {
      aspect-ratio: 1;
      font-size: 48px;
      font-weight: 800;
      border: 4px solid #e2e8f0;
      border-radius: var(--radius-md);
      background: #f1f5f9;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }
    .memory-card.selected { border-color: #f59e0b; background: #fff7ed; transform: scale(0.95); }
    .memory-card.matched { border-color: var(--success); background: #f0fdf4; color: var(--success); cursor: default; }

    /* Maze Styles */
    .maze-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 24px;
    }
    .maze-board {
      display: grid;
      gap: 4px;
      padding: 8px;
      background: #27272a;
      border-radius: 12px;
      border: 6px solid #18181b;
    }
    .maze-cell {
      width: 44px;
      height: 44px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 900;
      font-size: 18px;
    }
    .maze-wall { background: #18181b; color: #3f3f46; }
    .maze-path { background: #3f3f46; }
    .maze-start { background: var(--info); color: white; }
    .maze-goal { background: #f59e0b; color: white; }
    .maze-current { background: var(--success); color: white; box-shadow: 0 0 15px var(--success); z-index: 10; }
    
    .direction-pad {
      display: grid;
      grid-template-areas: ". up ." "left down right";
      gap: 12px;
    }
    .dir-btn {
      width: 92px;
      height: 92px;
      font-size: 32px;
      font-weight: 900;
      border-radius: 20px;
      border: none;
      background: #f1f5f9;
      color: #1e293b;
      cursor: pointer;
      box-shadow: 0 6px 0 #cbd5e1;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .dir-btn:active { transform: translateY(4px); box-shadow: 0 2px 0 #cbd5e1; }
    .dir-btn.up { grid-area: up; }
    .dir-btn.left { grid-area: left; }
    .dir-btn.down { grid-area: down; }
    .dir-btn.right { grid-area: right; }

    /* Arithmetic & Quiz Styles */
    .option-list { display: grid; gap: 16px; width: 100%; }
    .option-btn {
      min-height: 84px;
      font-size: 26px;
      font-weight: 800;
      border-radius: var(--radius-md);
      border: 3px solid #e2e8f0;
      background: white;
      color: var(--text);
      cursor: pointer;
      text-align: left;
      padding: 0 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: all 0.2s;
    }
    .option-btn:hover { border-color: var(--primary); background: #fff7ed; }
    .option-btn.selected { border-color: var(--primary); background: #fff7ed; box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.2); }
    .option-btn::after { content: '👉'; opacity: 0.3; }
    .option-btn.selected::after { content: '✅'; opacity: 1; }

    .numpad {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      max-width: 400px;
      margin: 20px auto;
    }
    .num-btn {
      height: 84px;
      font-size: 32px;
      font-weight: 900;
      border-radius: 18px;
      border: none;
      background: #f1f5f9;
      color: #1e293b;
      box-shadow: 0 6px 0 #cbd5e1;
    }
    .num-btn:active { transform: translateY(4px); box-shadow: 0 2px 0 #cbd5e1; }
    .num-btn.clear { background: #fee2e2; color: #b91c1c; box-shadow: 0 6px 0 #fecaca; }
    .num-btn.submit-small { background: #f0fdf4; color: #15803d; box-shadow: 0 6px 0 #bcf0da; }

    .footer-controls {
      display: flex;
      gap: 16px;
      margin-top: auto;
    }
    .btn-large {
      flex: 1;
      height: 80px;
      font-size: 26px;
      font-weight: 900;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
    }
    .btn-primary { background: var(--primary); color: white; box-shadow: 0 8px 20px rgba(249, 115, 22, 0.3); }
    .btn-secondary { background: #f1f5f9; color: var(--muted); border: 2px solid #e2e8f0; }
    .btn-primary:active, .btn-secondary:active { transform: translateY(2px); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    @media (max-width: 768px) {
      body { font-size: 20px; }
      .memory-grid { grid-template-columns: repeat(3, 1fr); }
      .hero-title { font-size: 28px; }
      .question-box { font-size: 32px; padding: 24px; }
      .dir-btn { width: 72px; height: 72px; font-size: 24px; }
    }
  </style>
</head>
<body>
  <script id="boot" type="application/json">__BOOT__</script>
  <div class="shell">
    <header class="hero">
      <div class="hero-top">
        <h1 class="hero-title" id="gameTitle">로딩 중...</h1>
        <div class="hero-info">
          <span class="pill" id="scoreText">점수: 0</span>
          <span class="pill" id="stageText">단 계: -</span>
        </div>
      </div>
      <div class="progress-container">
        <div class="progress-bar" id="progressBar" style="width: 0%"></div>
      </div>
    </header>

    <div class="layout">
      <section class="game-card">
        <div class="stage-header">
          <h2 class="stage-title" id="stageTitle">준비 중...</h2>
          <p class="stage-prompt" id="stagePrompt"></p>
        </div>

        <div id="feedbackBox" class="feedback">문제를 풀어보세요.</div>
        
        <div id="stageBody" class="board"></div>

        <div class="footer-controls">
          <button class="btn-large btn-secondary" id="resetBtn">처음부터</button>
          <button class="btn-large btn-primary" id="submitBtn">정답 제출하기</button>
        </div>
      </section>
    </div>
  </div>

  <script>
    const BOOT = JSON.parse(document.getElementById('boot').textContent);
    const API_ROOT = BOOT.apiOrigin;
    const state = {
      stage: null,
      progress: null,
      memory: { selected: [], matched: new Set() },
      maze: { path: '', x: 0, y: 0 },
      choice: '',
      history: []
    };

    const els = {
      gameTitle: document.getElementById('gameTitle'),
      scoreText: document.getElementById('scoreText'),
      stageText: document.getElementById('stageText'),
      progressBar: document.getElementById('progressBar'),
      stageTitle: document.getElementById('stageTitle'),
      stagePrompt: document.getElementById('stagePrompt'),
      feedbackBox: document.getElementById('feedbackBox'),
      stageBody: document.getElementById('stageBody'),
      submitBtn: document.getElementById('submitBtn'),
      resetBtn: document.getElementById('resetBtn')
    };

    function updateUi() {
      els.gameTitle.textContent = BOOT.game.title;
      els.scoreText.textContent = `점수: ${state.progress?.score || 0}`;
      
      const total = BOOT.game.totalStages || 1;
      const current = state.progress?.cleared ? total : (state.progress?.currentStageNo || 1);
      els.stageText.textContent = state.progress?.cleared ? '모두 완료!' : `단 계: ${current} / ${total}`;
      
      const progressPercent = Math.min(100, Math.round(((current - (state.progress?.cleared ? 0 : 1)) / total) * 100));
      els.progressBar.style.width = `${progressPercent}%`;

      if (state.progress?.cleared) {
        els.stageTitle.textContent = '축하합니다!';
        els.stagePrompt.textContent = '모든 문제를 다 풀었습니다.';
        els.stageBody.innerHTML = '<div class="question-box">🏅 참 잘하셨어요! 🏅</div>';
        showFeedback('오늘의 게임을 모두 완료하셨습니다!', 'ok');
        els.submitBtn.disabled = true;
        return;
      }

      if (!state.stage) return;

      els.stageTitle.textContent = state.stage.title;
      els.stagePrompt.textContent = state.stage.prompt;
      
      const lastMsg = state.progress?.state?.lastMessage;
      if (lastMsg) {
        showFeedback(lastMsg, state.progress.lastAnswerCorrect ? 'ok' : 'bad');
      } else {
        els.feedbackBox.classList.remove('visible');
      }

      renderStageContent();
    }

    function showFeedback(msg, type) {
      els.feedbackBox.textContent = msg;
      els.feedbackBox.className = `feedback visible ${type}`;
    }

    function renderStageContent() {
      const type = state.stage.stageType;
      const payload = state.stage.payload;

      if (type === 'memory_match') {
        els.stageBody.innerHTML = `
          <div class="memory-grid">
            ${payload.cards.map(card => {
              const matched = state.memory.matched.has(card.pairKey);
              const selected = state.memory.selected.includes(card.id);
              return `<button class="memory-card ${matched ? 'matched' : ''} ${selected ? 'selected' : ''}" 
                        data-id="${card.id}" data-key="${card.pairKey}" ${matched ? 'disabled' : ''}>
                        ${matched || selected ? card.label : '❓'}
                      </button>`;
            }).join('')}
          </div>
        `;
      } else if (type === 'maze') {
        const grid = payload.grid;
        els.stageBody.innerHTML = `
          <div class="maze-container">
            <div class="maze-board" style="grid-template-columns: repeat(${grid[0].length}, 1fr)">
              ${grid.map((row, y) => row.map((cell, x) => {
                const isCurrent = state.maze.x === x && state.maze.y === y;
                let cls = 'maze-cell';
                if (cell === '#') cls += ' maze-wall';
                else if (cell === 'S') cls += ' maze-start';
                else if (cell === 'G') cls += ' maze-goal';
                else cls += ' maze-path';
                if (isCurrent) cls += ' maze-current';
                return `<div class="${cls}">${isCurrent ? '🚶' : (cell === '#' ? '' : cell)}</div>`;
              }).join('')).join('')}
            </div>
            <div class="direction-pad">
              <button class="dir-btn up" data-move="U">▲</button>
              <button class="dir-btn left" data-move="L">◀</button>
              <button class="dir-btn down" data-move="D">▼</button>
              <button class="dir-btn right" data-move="R">▶</button>
            </div>
          </div>
        `;
      } else if (type === 'arithmetic') {
        els.stageBody.innerHTML = `
          <div class="question-box">${payload.question.replace('?', '___')}</div>
          <div class="question-box" style="background:#fff7ed; font-size: 54px; margin-top:0">${state.choice || '?'}</div>
          <div class="numpad">
            ${[1,2,3,4,5,6,7,8,9].map(n => `<button class="num-btn" data-val="${n}">${n}</button>`).join('')}
            <button class="num-btn clear" data-val="C">지우기</button>
            <button class="num-btn" data-val="0">0</button>
            <button class="num-btn submit-small" data-val="S">입력</button>
          </div>
        `;
      } else if (type === 'initials_quiz') {
        els.stageBody.innerHTML = `
          <div class="question-box" style="letter-spacing: 0.5em; font-size: 60px">${payload.clue}</div>
          <div class="option-list">
            ${payload.options.map((opt, i) => `
              <button class="option-btn ${state.choice === String(i) ? 'selected' : ''}" data-idx="${i}">
                ${i+1}. ${opt}
              </button>
            `).join('')}
          </div>
        `;
      }
    }

    async function api(path, method='GET', body=null) {
      const options = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) options.body = JSON.stringify(body);
      const res = await fetch(`${API_ROOT}${path}`, options);
      const json = await res.json();
      if (!res.ok || !json.success) throw new Error(json.message || '오류가 발생했습니다.');
      return json.data;
    }

    async function load() {
      const data = await api(`/api/v1/games/${BOOT.gameSlug}/state?userId=${BOOT.userId}`);
      state.game = data.game;
      state.progress = data.progress;
      state.stage = data.stage;
      resetLocal();
      updateUi();
    }

    function resetLocal() {
      state.memory = { selected: [], matched: new Set() };
      state.choice = '';
      if (state.stage?.stageType === 'maze') {
        state.maze = { path: '', x: state.stage.payload.start.x, y: state.stage.payload.start.y };
      }
    }

    els.stageBody.onclick = (e) => {
      const btn = e.target.closest('button, [data-move]');
      if (!btn || state.progress?.cleared) return;

      if (btn.dataset.id) { // Memory
        const id = btn.dataset.id;
        const key = btn.dataset.key;
        if (state.memory.matched.has(key) || state.memory.selected.includes(id)) return;
        state.memory.selected.push(id);
        if (state.memory.selected.length === 2) {
          const cards = state.stage.payload.cards;
          const c1 = cards.find(c => c.id === state.memory.selected[0]);
          const c2 = cards.find(c => c.id === state.memory.selected[1]);
          if (c1.pairKey === c2.pairKey) {
            state.memory.matched.add(c1.pairKey);
            showFeedback('정답입니다! 짝을 맞췄어요.', 'ok');
          } else {
            showFeedback('틀렸습니다. 다시 해볼까요?', 'bad');
          }
          setTimeout(() => { state.memory.selected = []; renderStageContent(); }, 1000);
        }
      } else if (btn.dataset.move) { // Maze
        const mv = btn.dataset.move;
        const grid = state.stage.payload.grid;
        let nx = state.maze.x, ny = state.maze.y;
        if (mv === 'U') ny--; else if (mv === 'D') ny++; else if (mv === 'L') nx--; else if (mv === 'R') nx++;
        if (ny >= 0 && ny < grid.length && nx >= 0 && nx < grid[0].length && grid[ny][nx] !== '#') {
          state.maze.x = nx; state.maze.y = ny; state.maze.path += mv;
          if (grid[ny][nx] === 'G') showFeedback('출구에 도착했습니다! 제출 버튼을 누르세요.', 'info');
        }
      } else if (btn.dataset.val) { // Arithmetic Numpad
        const val = btn.dataset.val;
        if (val === 'C') state.choice = '';
        else if (val === 'S') submit();
        else state.choice += val;
      } else if (btn.dataset.idx) { // Choice
        state.choice = btn.dataset.idx;
      }
      renderStageContent();
    };

    async function submit() {
      let answer = null;
      const type = state.stage.stageType;
      if (type === 'memory_match') answer = { matchedPairs: Array.from(state.memory.matched) };
      else if (type === 'maze') answer = { path: state.maze.path };
      else if (type === 'arithmetic') {
        const val = state.choice;
        const options = state.stage.payload.options;
        const idx = options.findIndex(o => String(o) === val);
        answer = { value: idx !== -1 ? String(idx) : val };
      }
      else if (type === 'initials_quiz') answer = { value: state.choice };

      if (!answer) return;
      try {
        const data = await api(`/api/v1/games/${BOOT.gameSlug}/answer`, 'POST', {
          userId: BOOT.userId, gameSlug: BOOT.gameSlug, stageNo: state.stage.stageNo, answer
        });
        state.progress = data.progress;
        state.stage = data.stage;
        resetLocal();
        updateUi();
      } catch (e) { showFeedback(e.message, 'bad'); }
    }

    els.submitBtn.onclick = submit;
    els.resetBtn.onclick = async () => {
      if (!confirm('처음부터 다시 시작하시겠습니까?')) return;
      const data = await api(`/api/v1/games/${BOOT.gameSlug}/reset`, 'POST', {
        userId: BOOT.userId, gameSlug: BOOT.gameSlug
      });
      state.progress = data.progress;
      state.stage = data.stage;
      resetLocal();
      updateUi();
    };

    load();
  </script>
</body>
</html>
"""

        html = (
            template.replace("__BOOT__", _to_json(boot))
            .replace("__TITLE__", escape(catalog.title))
            .replace("__PRIMARY__", escape(catalog.theme_color))
        )
        return HTMLResponse(content=html)
