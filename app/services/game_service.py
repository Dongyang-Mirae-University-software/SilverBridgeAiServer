from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
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


def _maze_reaches_goal(grid: list[list[str]], path: str) -> bool:
    """제출한 이동 경로가 벽을 뚫지 않고 S에서 G까지 가는지 확인한다."""
    if not grid:
        return False
    start = goal = None
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "S":
                start = (x, y)
            elif cell == "G":
                goal = (x, y)
    if start is None or goal is None:
        return False
    x, y = start
    for move in _normalize_dir_path(path):
        dx, dy = {"R": (1, 0), "L": (-1, 0), "D": (0, 1), "U": (0, -1)}[move]
        x, y = x + dx, y + dy
        if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[y]) or grid[y][x] == "#":
            return False
    return (x, y) == goal


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


def _maze_stage(stage_no: int, path: str, size: int = 6) -> StageSeed:
    grid, start, goal, moves = _maze_grid_from_path(path, size=size)
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


STAGES_PER_GAME = 50

_CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"

MEMORY_WORDS = [
    "사과", "별", "고양이", "바다", "하늘", "꽃", "달", "해", "나무", "물", "기차", "버스", "자동차", "자전거",
    "연필", "지우개", "공책", "가방", "책", "의자", "책상", "램프", "창문", "문", "토끼", "사자", "호랑이", "기린",
    "코끼리", "서울", "부산", "대구", "광주", "대전", "울산", "수박", "포도", "딸기", "감자", "고구마", "우산",
    "모자", "신발", "안경", "시계", "전화", "라디오", "냄비", "숟가락", "젓가락", "거울", "비누", "수건", "이불",
]

# 초성퀴즈 단어 풀: 2~3음절 일상 명사. 정답과 보기는 여기서 뽑고, 보기는 초성이 정답과 다른 단어만 쓴다.
INITIALS_WORDS = [
    "가방", "기차", "고기", "기분", "가족", "감자", "거울", "구름", "김치", "국수", "나무", "나비", "냄비", "노래",
    "눈물", "다리", "달걀", "도시", "돼지", "두부", "라디오", "마늘", "모자", "무지개", "바다", "버스", "방울", "배달",
    "비누", "사과", "시계", "소금", "수박", "신발", "아기", "안경", "우유", "의자", "오이", "연필", "우산", "은행",
    "자동차", "자전거", "지갑", "장미", "전화", "초밥", "책상", "출발", "차별", "치즈", "창문", "카메라", "커피",
    "코끼리", "택시", "토마토", "튤립", "파도", "편지", "포도", "하늘", "학교", "호수", "호박", "휴지", "고양이",
    "강아지", "호랑이", "기린", "다람쥐", "병원", "약국", "시장", "공원", "극장", "식당", "부엌", "화장실", "정원",
    "김밥", "라면", "만두", "된장", "간장", "설탕", "후추", "딸기", "참외", "복숭아", "당근", "양파", "상추", "배추",
    "손목", "무릎", "어깨", "허리", "얼굴", "머리", "손가락", "발가락", "지우개", "공책", "숟가락", "젓가락",
    "수건", "이불", "베개", "장갑", "목도리", "치마", "바지", "양말", "구두", "단추", "바늘", "가위", "종이",
]


def _initials(word: str) -> str:
    result = []
    for ch in word:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            result.append(_CHOSUNG[(code - 0xAC00) // 588])
        else:
            result.append(ch)
    return "".join(result)


def _random_walk(rng: random.Random, size: int, length: int) -> str:
    """(0,0)에서 시작하는 자기회피 랜덤 워크. 실패하면 다시 시도한다."""
    while True:
        x = y = 0
        visited = {(0, 0)}
        moves: list[str] = []
        ok = True
        for _ in range(length):
            options = []
            for move, dx, dy in (("R", 1, 0), ("L", -1, 0), ("D", 0, 1), ("U", 0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                    options.append((move, nx, ny))
            if not options:
                ok = False
                break
            move, x, y = rng.choice(options)
            visited.add((x, y))
            moves.append(move)
        if ok:
            return "".join(moves)


def _generate_memory_stages() -> list[StageSeed]:
    seeds = []
    for n in range(1, STAGES_PER_GAME + 1):
        rng = random.Random(9000 + n)
        pairs = 3 + (n - 1) // 10  # 3 → 7쌍
        seeds.append(_memory_stage(n, rng.sample(MEMORY_WORDS, pairs)))
    return seeds


def _generate_maze_stages() -> list[StageSeed]:
    seeds = []
    for n in range(1, STAGES_PER_GAME + 1):
        rng = random.Random(8000 + n)
        size = 6 if n <= 25 else 7
        length = 4 + (n - 1) // 4  # 4 → 16칸
        seeds.append(_maze_stage(n, _random_walk(rng, size, length), size))
    return seeds


def _generate_arithmetic_stages() -> list[StageSeed]:
    seeds = []
    for n in range(1, STAGES_PER_GAME + 1):
        rng = random.Random(6000 + n)
        if n <= 15:
            op = rng.choice(["+", "-"])
            left, right = rng.randint(1, 10), rng.randint(1, 10)
        elif n <= 30:
            op = rng.choice(["+", "-", "×"])
            if op == "×":
                left, right = rng.randint(2, 9), rng.randint(2, 9)
            else:
                left, right = rng.randint(10, 60), rng.randint(1, 40)
        else:
            op = rng.choice(["+", "-", "×", "÷"])
            if op == "×":
                left, right = rng.randint(2, 12), rng.randint(2, 9)
            elif op == "÷":
                right = rng.randint(2, 9)
                left = right * rng.randint(2, 12)
            else:
                left, right = rng.randint(20, 99), rng.randint(1, 50)
        if op == "-" and left < right:
            left, right = right, left
        correct = {"+": left + right, "-": left - right, "×": left * right, "÷": left // right}[op]
        options = {correct}
        while len(options) < 4:
            candidate = correct + rng.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10])
            if candidate >= 0:
                options.add(candidate)
        option_list = list(options)
        rng.shuffle(option_list)
        seeds.append(_arithmetic_stage(n, left, op, right, option_list))
    return seeds


def _generate_initials_stages() -> list[StageSeed]:
    pool_rng = random.Random(7000)
    answers = pool_rng.sample(INITIALS_WORDS, STAGES_PER_GAME)
    seeds = []
    for n, answer in enumerate(answers, start=1):
        rng = random.Random(7000 + n)
        clue = _initials(answer)
        # 보기는 초성이 정답과 다른 단어만. 후반부는 첫 초성이 같은 단어를 우선 섞어 난이도를 올린다.
        candidates = [w for w in INITIALS_WORDS if w != answer and _initials(w) != clue]
        if n > 25:
            similar = [w for w in candidates if w[0] and _initials(w)[0] == clue[0]]
            rng.shuffle(similar)
            rng.shuffle(candidates)
            distractors = (similar + [w for w in candidates if w not in similar])[:3]
        else:
            distractors = rng.sample(candidates, 3)
        options = distractors + [answer]
        rng.shuffle(options)
        seeds.append(_initials_stage(n, clue, answer, options))
    return seeds


def build_game_seed_data() -> list[StageSeed]:
    seeds: list[StageSeed] = []
    seeds.extend(_generate_memory_stages())
    seeds.extend(_generate_maze_stages())
    seeds.extend(_generate_arithmetic_stages())
    seeds.extend(_generate_initials_stages())
    return seeds


class GameService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _ensure_seeded(self) -> None:
        with SEED_LOCK:
            if self.db.query(GameCatalog).count() > 0:
                self._reseed_stages_if_changed()
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

    def _reseed_stages_if_changed(self) -> None:
        """시드 문제 수가 바뀌면(예: 8개 → 50개) 스테이지만 갈아끼운다. 진행 정보는 유지."""
        stages = build_game_seed_data()
        expected: dict[str, int] = {}
        for seed in stages:
            expected[seed.game_slug] = expected.get(seed.game_slug, 0) + 1
        catalogs = {row.slug: row for row in self.db.query(GameCatalog).all()}
        if all(catalog.total_stages == expected.get(slug, 0) for slug, catalog in catalogs.items()):
            return
        self.db.query(GameStage).delete()
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
        for slug, catalog in catalogs.items():
            catalog.total_stages = expected.get(slug, 0)
            self.db.add(catalog)
        for progress in self.db.query(GameProgress).all():
            total = expected.get(progress.game_slug, 0)
            if not progress.cleared and progress.current_stage_no > total:
                progress.cleared = True
                progress.cleared_at = _now()
                self.db.add(progress)
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
            message = "모든 짝을 맞췄습니다." if correct else f"{len(required_pairs)}쌍 중 {len(submitted_pairs & required_pairs)}쌍을 맞췄어요."
        elif stage_type == "maze":
            submitted_path = _normalize_dir_path(answer.get("path") if isinstance(answer, dict) else answer)
            payload = _from_json(stage.payload_json, {})
            correct = _maze_reaches_goal(payload.get("grid") or [], submitted_path)
            message = "출구에 도착했습니다." if correct else "출구에 도착하지 못했어요."
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
            message = "정답입니다." if correct else f"아쉬워요. 정답은 {expected_value}입니다."
        elif stage_type == "initials_quiz":
            submitted_value = _normalize_option(answer)
            expected_value = _normalize_answer_value(expected.get("value"))
            payload = _from_json(stage.payload_json, {})
            options = payload.get("options") or []
            clue = str(payload.get("clue") or "")
            if submitted_value.isdigit() and 0 <= int(submitted_value) < len(options):
                submitted_value = _normalize_answer_value(options[int(submitted_value)])
            # 초성이 힌트와 일치하는 보기는 모두 정답으로 인정한다(보기 중복 방어).
            correct = submitted_value == expected_value or (bool(clue) and _initials(submitted_value) == clue)
            message = "정답입니다." if correct else f"아쉬워요. 정답은 '{expected_value}'입니다."
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

        # 정오답과 상관없이 다음 문제로 넘어가고, 맞힌 문제만 점수가 쌓인다.
        if correct:
            progress.score += score_delta
            message = f"{message} +{score_delta}점"
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
        # 처음부터: 문제 진행만 1단계로 되돌리고 누적 점수·시도 횟수는 유지한다.
        progress.current_stage_no = 1
        progress.cleared = False
        progress.last_answer_correct = None
        progress.cleared_at = None
        progress.state_json = _to_json(
            {"currentStageNo": 1, "score": progress.score, "attempts": progress.attempts, "cleared": False},
        )
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

    def list_progress_for_user(self, user_id: int) -> dict[str, Any]:
        self._ensure_seeded()
        catalogs = self.db.query(GameCatalog).order_by(GameCatalog.id.asc()).all()
        rows = {row.game_slug: row for row in self.db.query(GameProgress).filter(GameProgress.user_id == user_id).all()}
        games = [
            {
                "game": self._catalog_payload(catalog),
                "progress": self._progress_payload(rows[catalog.slug]) if catalog.slug in rows else None,
            }
            for catalog in catalogs
        ]
        return {
            "userId": user_id,
            "totalScore": sum(row.score for row in rows.values()),
            "totalAttempts": sum(row.attempts for row in rows.values()),
            "lastPlayedAt": _iso(max((row.updated_at for row in rows.values()), default=None)),
            "games": games,
        }

    def list_activity_for_user(self, user_id: int, days: int = 182) -> dict[str, Any]:
        """날짜(KST)별 풀이 횟수·정답 수·획득 점수. 활동이 있는 날만 돌려준다."""
        days = max(1, min(days, 366))
        since = _now() - timedelta(days=days)
        rows = (
            self.db.query(GameAttempt)
            .filter(GameAttempt.user_id == user_id, GameAttempt.created_at >= since)
            .order_by(GameAttempt.created_at.asc())
            .all()
        )
        by_date: dict[str, dict[str, int]] = {}
        for row in rows:
            key = (row.created_at + timedelta(hours=9)).date().isoformat()
            bucket = by_date.setdefault(key, {"attempts": 0, "correct": 0, "score": 0})
            bucket["attempts"] += 1
            bucket["correct"] += 1 if row.is_correct else 0
            bucket["score"] += row.score_delta
        return {
            "userId": user_id,
            "days": days,
            "activity": [{"date": key, **value} for key, value in sorted(by_date.items())],
        }

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
    html { overflow: hidden; }
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
      padding: 12px;
      gap: 12px;
    }
    header.hero {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 12px 20px;
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
      font-size: 26px;
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
      height: 12px;
      background: #fed7aa;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 4px;
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
      padding: 16px 20px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 12px;
      flex: 1;
    }
    .stage-header { text-align: center; }
    .stage-title { font-size: 24px; font-weight: 800; margin: 0 0 4px; }
    .stage-prompt { font-size: 18px; color: var(--muted); margin: 0; font-weight: 500; }
    
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
      font-size: 20px;
      font-weight: 800;
      padding: 12px;
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
      margin-top: 0;
    }
    .memory-card {
      aspect-ratio: 1;
      min-height: 0;
      overflow: hidden;
      word-break: keep-all;
      font-size: calc(var(--card-size, 160px) * 0.28);
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
      height: 60px;
      font-size: 22px;
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

    function renderUi() {
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
      fitToViewport();
    }

    // iframe·태블릿처럼 작은 화면에서도 스크롤 없이 한 화면에 들어가도록 전체를 축소한다.
    // 짝맞추기 카드는 aspect-ratio라 높이가 가로폭을 따라간다(zoom으로는 안 줄어듦).
    // 그리드를 뺀 나머지 높이를 재서 남는 공간에 맞게 카드 한 변을 px로 고정한다.
    function fitMemoryGrid(shell) {
      const grid = document.querySelector('.memory-grid');
      if (!grid) return;
      grid.style.display = 'none';
      shell.style.minHeight = '0';
      // documentElement.scrollHeight는 뷰포트보다 작아지지 않으므로 shell 실제 높이를 잰다.
      const usedHeight = shell.offsetHeight;
      grid.style.display = '';
      grid.style.gridTemplateColumns = '';
      const gap = parseFloat(getComputedStyle(grid).gap) || 16;
      const count = grid.children.length;
      const availHeight = window.innerHeight - usedHeight;
      const availWidth = grid.clientWidth;
      // 카드가 가장 커지는 열 수를 고른다(넓은 화면이면 한 줄, 좁으면 여러 줄).
      let cols = 1;
      let size = 0;
      for (let c = 1; c <= count; c++) {
        const rows = Math.ceil(count / c);
        const fit = Math.min((availWidth - gap * (c - 1)) / c, (availHeight - gap * (rows - 1)) / rows);
        if (fit > size) { size = fit; cols = c; }
      }
      const cardSize = Math.max(48, Math.floor(size));
      grid.style.gridTemplateColumns = `repeat(${cols}, ${cardSize}px)`;
      grid.style.setProperty('--card-size', `${cardSize}px`);
      grid.style.justifyContent = 'center';
    }

    function fitToViewport() {
      const shell = document.querySelector('.shell');
      document.body.style.zoom = '';
      shell.style.minHeight = '0';
      fitMemoryGrid(shell);
      const contentHeight = shell.offsetHeight;
      const zoom = contentHeight > window.innerHeight ? window.innerHeight / contentHeight : 1;
      document.body.style.zoom = zoom;
      shell.style.minHeight = `${window.innerHeight / zoom}px`;
    }

    function updateUi() {
      renderUi();
      fitToViewport();
    }
    window.addEventListener('resize', fitToViewport);

    function showFeedback(msg, type) {
      els.feedbackBox.textContent = msg;
      els.feedbackBox.className = `feedback visible ${type}`;
      fitToViewport();
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
      let json;
      try {
        json = await res.json();
      } catch {
        // nginx 오류 페이지(HTML) 등 JSON이 아닌 응답: 서버 재시작 중이거나 연결 실패
        throw new Error(`서버에 연결할 수 없습니다 (${res.status}). 잠시 후 다시 시도해주세요.`);
      }
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
        // 틀린 두 장이 다시 뒤집히길 기다리는 동안(선택 2장)에는 다른 카드를 열 수 없다.
        if (state.memory.selected.length >= 2) return;
        if (state.memory.matched.has(key) || state.memory.selected.includes(id)) return;
        state.memory.selected.push(id);
        if (state.memory.selected.length === 2) {
          const cards = state.stage.payload.cards;
          const c1 = cards.find(c => c.id === state.memory.selected[0]);
          const c2 = cards.find(c => c.id === state.memory.selected[1]);
          if (c1.pairKey === c2.pairKey) {
            state.memory.matched.add(c1.pairKey);
            state.memory.selected = [];
            showFeedback('정답입니다! 짝을 맞췄어요.', 'ok');
          } else {
            showFeedback('틀렸습니다. 다시 해볼까요?', 'bad');
            setTimeout(() => { state.memory.selected = []; renderStageContent(); fitToViewport(); }, 1000);
          }
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
      fitToViewport();
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
      if (!confirm('1번 문제부터 다시 시작할까요? (지금까지 쌓은 점수는 그대로 남아요)')) return;
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
