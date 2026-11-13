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
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css" />
  <style>
    /* 어르신용: 큰 글씨·큰 버튼, 높은 대비(본문 대비 12:1 이상), 차분하고 밝은 색 */
    :root {
      --bg: #f6f3ee;
      --surface: #ffffff;
      --text: #1d2733;
      --muted: #4a5563;
      --line: #e4ded4;
      --primary: #2f5d8a;
      --primary-soft: #e8eff6;
      --ok: #1f7a4d;
      --ok-soft: #e6f4ec;
      --try: #8a5a00;
      --try-soft: #fdf3dc;
      --radius-lg: 28px;
      --radius-md: 20px;
      --shadow: 0 2px 4px rgba(29, 39, 51, 0.04), 0 8px 24px rgba(29, 39, 51, 0.06);
    }
    /* 게임별 색. 흰 글씨와 대비 4.5:1 이상이 되도록 고른 진한 색 */
    body[data-game="memory_match"] { --primary: #5b45b0; --primary-soft: #efebfa; }
    body[data-game="maze"] { --primary: #0b6e96; --primary-soft: #e3f2f8; }
    body[data-game="arithmetic"] { --primary: #b04a08; --primary-soft: #fdeee3; }
    body[data-game="initials_quiz"] { --primary: #2b7049; --primary-soft: #e6f3eb; }

    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    html { overflow: hidden; }
    body {
      margin: 0;
      font-family: "Pretendard Variable", Pretendard, system-ui, -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 22px;
      line-height: 1.45;
      word-break: keep-all;
    }
    button { font: inherit; color: inherit; }
    button:focus-visible { outline: 4px solid var(--primary); outline-offset: 3px; }
    .shell { min-height: 100vh; display: flex; flex-direction: column; gap: 14px; padding: 14px; }

    /* 위쪽: 제목·점수·소리 */
    .topbar {
      display: flex; flex-direction: column; gap: 12px;
      padding: 16px 20px; background: var(--surface);
      border-radius: var(--radius-lg); box-shadow: var(--shadow);
    }
    .topbar-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    .title-wrap { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
    .game-title { margin: 0; font-size: 28px; font-weight: 800; color: var(--primary); letter-spacing: -0.02em; }
    .stage-count { font-size: 20px; font-weight: 700; color: var(--muted); }
    .top-actions { display: flex; gap: 10px; align-items: center; }
    .score {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 8px 16px; border-radius: 999px;
      background: var(--primary-soft); color: var(--primary);
      font-size: 20px; font-weight: 800;
    }
    .sound-btn {
      min-height: 48px; padding: 0 16px; border-radius: 999px;
      border: 2px solid var(--line); background: var(--surface);
      font-size: 18px; font-weight: 700; color: var(--muted); cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .sound-btn[aria-pressed="true"] { color: var(--text); border-color: var(--text); }
    .progress { height: 10px; background: var(--line); border-radius: 999px; overflow: hidden; }
    .progress-bar { height: 100%; width: 0; background: var(--primary); border-radius: 999px; transition: width 0.5s ease; }

    /* 가운데: 문제 카드 */
    .game-card {
      flex: 1; display: flex; flex-direction: column; gap: 16px;
      padding: 20px; background: var(--surface);
      border-radius: var(--radius-lg); box-shadow: var(--shadow);
    }
    .prompt { margin: 0; text-align: center; font-size: 26px; font-weight: 800; letter-spacing: -0.01em; }
    .feedback {
      display: none; align-items: center; justify-content: center; gap: 10px;
      padding: 14px 18px; border-radius: var(--radius-md);
      font-size: 22px; font-weight: 700; text-align: center;
    }
    .feedback.visible { display: flex; }
    .feedback.ok { background: var(--ok-soft); color: var(--ok); }
    .feedback.try { background: var(--try-soft); color: var(--try); }
    .feedback.info { background: var(--primary-soft); color: var(--primary); }
    .board { width: 100%; display: flex; flex-direction: column; align-items: center; gap: 16px; }

    /* 짝 맞추기 */
    .memory-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; width: 100%; }
    .memory-card {
      aspect-ratio: 1; min-height: 0; overflow: hidden; padding: 6px;
      font-size: calc(var(--card-size, 140px) * 0.24); font-weight: 800;
      border: 3px solid transparent; border-radius: var(--radius-md); cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      background: var(--primary); color: #fff;
      transition: transform 0.2s ease, background 0.2s ease;
    }
    .memory-card .back-mark {
      width: 46%; aspect-ratio: 1; border-radius: 50%;
      border: 3px solid rgba(255, 255, 255, 0.55);
      display: flex; align-items: center; justify-content: center;
      font-size: calc(var(--card-size, 140px) * 0.22);
    }
    .memory-card.open { background: var(--surface); color: var(--text); border-color: var(--primary); transform: scale(0.97); }
    .memory-card.matched { background: var(--ok-soft); color: var(--ok); border-color: var(--ok); cursor: default; }

    /* 미로 찾기 */
    .maze-board { display: grid; gap: 6px; padding: 10px; background: #e9e4da; border-radius: 22px; }
    .maze-cell {
      width: 56px; height: 56px; border-radius: 12px; border: none; padding: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 30px; font-weight: 800;
    }
    .maze-wall { background: #a8b0bb; }
    .maze-path { background: var(--surface); }
    .maze-next { background: var(--primary-soft); cursor: pointer; box-shadow: inset 0 0 0 3px var(--primary); }
    .maze-goal { background: #ffe8a3; }
    .maze-me { background: var(--primary); color: #fff; }
    .direction-pad { display: grid; grid-template-areas: ". up ." "left down right"; gap: 12px; }
    .dir-btn {
      width: 96px; height: 84px; border-radius: 22px; border: 2px solid var(--line);
      background: var(--surface); cursor: pointer;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      font-size: 30px; font-weight: 800; line-height: 1.1; box-shadow: 0 4px 0 var(--line);
    }
    .dir-btn small { font-size: 16px; font-weight: 700; color: var(--muted); }
    .dir-btn:active { transform: translateY(3px); box-shadow: 0 1px 0 var(--line); }
    .dir-btn.up { grid-area: up; } .dir-btn.left { grid-area: left; }
    .dir-btn.down { grid-area: down; } .dir-btn.right { grid-area: right; }

    /* 사칙연산·초성 퀴즈 */
    .question {
      width: 100%; padding: 28px 16px; text-align: center;
      background: var(--primary-soft); border-radius: var(--radius-md);
      font-size: 56px; font-weight: 800; letter-spacing: -0.01em; color: var(--text);
    }
    .question .blank {
      display: inline-block; min-width: 1.6em; padding: 0 0.2em;
      border-bottom: 5px solid var(--primary); color: var(--primary);
    }
    .tiles { display: flex; justify-content: center; gap: 14px; }
    .tile {
      width: 104px; height: 104px; border-radius: 22px; background: var(--surface);
      border: 3px solid var(--primary); color: var(--primary);
      display: flex; align-items: center; justify-content: center; font-size: 60px; font-weight: 800;
    }
    .options { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; width: 100%; }
    .option-btn {
      min-height: 92px; padding: 10px 16px; border-radius: var(--radius-md);
      border: 3px solid var(--line); background: var(--surface);
      font-size: 34px; font-weight: 800; cursor: pointer;
      transition: border-color 0.15s ease, background 0.15s ease;
    }
    .option-btn.selected { border-color: var(--primary); background: var(--primary-soft); color: var(--primary); box-shadow: 0 0 0 4px var(--primary-soft); }

    /* 완료 화면 */
    .done { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 24px 0; text-align: center; }
    .medal {
      width: 140px; height: 140px; border-radius: 50%;
      background: #ffe8a3; display: flex; align-items: center; justify-content: center; font-size: 72px;
    }
    .done strong { font-size: 34px; }
    .done span { font-size: 22px; color: var(--muted); }

    /* 아래: 버튼 */
    .footer { display: flex; gap: 12px; margin-top: auto; }
    .btn {
      min-height: 72px; padding: 0 24px; border-radius: 999px; border: none; cursor: pointer;
      font-size: 24px; font-weight: 800;
      display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    .btn-primary { flex: 2; background: var(--primary); color: #fff; }
    .btn-secondary { flex: 1; background: var(--surface); color: var(--muted); border: 2px solid var(--line); }
    .btn:active { transform: translateY(2px); }
    .btn-primary:disabled { background: #c9ced6; color: #4a5563; cursor: not-allowed; transform: none; }

    @media (max-width: 640px) {
      body { font-size: 20px; }
      .game-title { font-size: 24px; }
      .question { font-size: 44px; padding: 20px 12px; }
      .tile { width: 80px; height: 80px; font-size: 46px; }
      .option-btn { font-size: 28px; min-height: 80px; }
      .maze-cell { width: 46px; height: 46px; font-size: 24px; }
      .dir-btn { width: 84px; height: 76px; }
    }
    @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  </style>
</head>
<body>
  <script id="boot" type="application/json">__BOOT__</script>
  <div class="shell">
    <header class="topbar">
      <div class="topbar-row">
        <div class="title-wrap">
          <h1 class="game-title" id="gameTitle">불러오는 중</h1>
          <span class="stage-count" id="stageText"></span>
        </div>
        <div class="top-actions">
          <span class="score" id="scoreText">⭐ 0점</span>
          <button type="button" class="sound-btn" id="soundBtn" aria-pressed="true">🔊 소리 켜짐</button>
        </div>
      </div>
      <div class="progress" aria-hidden="true"><div class="progress-bar" id="progressBar"></div></div>
    </header>

    <main class="game-card">
      <h2 class="prompt" id="stagePrompt">잠시만 기다려 주세요</h2>
      <div id="feedbackBox" class="feedback" role="status" aria-live="polite"></div>
      <div id="stageBody" class="board"></div>
      <div class="footer">
        <button type="button" class="btn btn-secondary" id="resetBtn">처음부터</button>
        <button type="button" class="btn btn-primary" id="submitBtn">확인</button>
      </div>
    </main>
  </div>

  <script>
    const BOOT = JSON.parse(document.getElementById('boot').textContent);
    const API_ROOT = BOOT.apiOrigin;
    document.body.dataset.game = BOOT.gameSlug;
    const state = {
      stage: null,
      progress: null,
      memory: { selected: [], matched: new Set() },
      maze: { path: '', x: 0, y: 0 },
      choice: '',
      busy: false
    };
    const els = {
      gameTitle: document.getElementById('gameTitle'),
      scoreText: document.getElementById('scoreText'),
      stageText: document.getElementById('stageText'),
      progressBar: document.getElementById('progressBar'),
      stagePrompt: document.getElementById('stagePrompt'),
      feedbackBox: document.getElementById('feedbackBox'),
      stageBody: document.getElementById('stageBody'),
      submitBtn: document.getElementById('submitBtn'),
      resetBtn: document.getElementById('resetBtn'),
      soundBtn: document.getElementById('soundBtn')
    };

    // 소리·진동: 버튼 하나로 함께 켜고 끈다. 설정은 이 기기에만 저장(저장이 막혀도 동작)
    let soundOn = true;
    try { soundOn = localStorage.getItem('sb-game-sound') !== 'off'; } catch (e) {}
    let audioCtx = null;
    function renderSoundBtn() {
      els.soundBtn.setAttribute('aria-pressed', String(soundOn));
      els.soundBtn.textContent = soundOn ? '🔊 소리 켜짐' : '🔇 소리 꺼짐';
    }
    els.soundBtn.onclick = () => {
      soundOn = !soundOn;
      try { localStorage.setItem('sb-game-sound', soundOn ? 'on' : 'off'); } catch (e) {}
      renderSoundBtn();
      if (soundOn) chime('tap');
    };
    function chime(kind) {
      if (!soundOn) return;
      try {
        audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        const notes = kind === 'ok' ? [523, 659, 784] : kind === 'try' ? [440, 392] : [660];
        notes.forEach((freq, i) => {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          const t = audioCtx.currentTime + i * 0.14;
          osc.type = 'sine';
          osc.frequency.value = freq;
          gain.gain.setValueAtTime(0.0001, t);
          gain.gain.exponentialRampToValueAtTime(0.18, t + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
          osc.connect(gain).connect(audioCtx.destination);
          osc.start(t);
          osc.stop(t + 0.32);
        });
      } catch (e) {}
      if (kind === 'ok' && navigator.vibrate) { try { navigator.vibrate(40); } catch (e) {} }
    }

    function showFeedback(msg, type) {
      els.feedbackBox.textContent = msg;
      els.feedbackBox.className = `feedback visible ${type}`;
      fitToViewport();
    }
    function hideFeedback() { els.feedbackBox.className = 'feedback'; }

    function stageReady() {
      const type = state.stage?.stageType;
      if (type === 'memory_match') return state.memory.matched.size === state.stage.payload.pairCount;
      if (type === 'maze') return state.stage.payload.grid[state.maze.y][state.maze.x] === 'G';
      return state.choice !== '';
    }

    function renderUi() {
      els.gameTitle.textContent = BOOT.game.title;
      els.scoreText.textContent = `⭐ ${state.progress?.score || 0}점`;
      const total = BOOT.game.totalStages || 1;
      const cleared = !!state.progress?.cleared;
      const current = cleared ? total : (state.progress?.currentStageNo || 1);
      els.stageText.textContent = cleared ? '모두 마쳤어요' : `${current}번째 / ${total}`;
      els.progressBar.style.width = `${Math.min(100, Math.round(((cleared ? total : current - 1) / total) * 100))}%`;

      if (cleared) {
        els.stagePrompt.textContent = '오늘도 수고 많으셨어요';
        els.stageBody.innerHTML = `<div class="done"><div class="medal" aria-hidden="true">🏅</div>
          <strong>참 잘하셨어요!</strong><span>모든 문제를 다 풀었어요. 모은 점수는 ${state.progress.score}점이에요.</span></div>`;
        hideFeedback();
        els.submitBtn.hidden = true;
        els.resetBtn.textContent = '처음부터 다시';
        fitToViewport();
        return;
      }
      els.submitBtn.hidden = false;
      els.resetBtn.textContent = '처음부터';
      if (!state.stage) return;
      els.stagePrompt.textContent = state.stage.prompt;
      const lastMsg = state.progress?.state?.lastMessage;
      if (lastMsg) showFeedback(`앞 문제: ${lastMsg}`, state.progress.lastAnswerCorrect ? 'ok' : 'try');
      else hideFeedback();
      renderStageContent();
    }

    function renderStageContent() {
      const type = state.stage.stageType;
      const payload = state.stage.payload;
      // 짝 맞추기·미로는 다 하면 저절로 넘어가므로, 버튼은 '넘어가기' 역할만 한다
      const autoGame = type === 'memory_match' || type === 'maze';
      els.submitBtn.textContent = autoGame ? '다음 문제로' : '확인';
      els.submitBtn.disabled = state.busy || (!autoGame && state.choice === '');

      if (type === 'memory_match') {
        els.stageBody.innerHTML = `<div class="memory-grid">${payload.cards.map(card => {
          const matched = state.memory.matched.has(card.pairKey);
          const open = matched || state.memory.selected.includes(card.id);
          const cls = matched ? 'matched' : (open ? 'open' : '');
          const label = open ? card.label : '<span class="back-mark" aria-hidden="true">?</span>';
          return `<button type="button" class="memory-card ${cls}" data-id="${card.id}" data-key="${card.pairKey}"
            aria-label="${open ? card.label : '뒤집힌 카드'}" ${matched ? 'disabled' : ''}>${label}</button>`;
        }).join('')}</div>`;
      } else if (type === 'maze') {
        const grid = payload.grid;
        const { x: mx, y: my } = state.maze;
        els.stageBody.innerHTML = `
          <div class="maze-board" style="grid-template-columns: repeat(${grid[0].length}, auto)">
            ${grid.map((row, y) => row.map((cell, x) => {
              const me = x === mx && y === my;
              const near = Math.abs(x - mx) + Math.abs(y - my) === 1 && cell !== '#';
              let cls = cell === '#' ? 'maze-wall' : cell === 'G' ? 'maze-goal' : 'maze-path';
              if (near) cls += ' maze-next';
              if (me) cls = 'maze-me';
              const move = x > mx ? 'R' : x < mx ? 'L' : y > my ? 'D' : 'U';
              const icon = me ? '🙂' : cell === 'G' ? '🏁' : '';
              return near
                ? `<button type="button" class="maze-cell ${cls}" data-move="${move}" aria-label="이 칸으로 가기">${icon}</button>`
                : `<div class="maze-cell ${cls}">${icon}</div>`;
            }).join('')).join('')}
          </div>
          <div class="direction-pad">
            <button type="button" class="dir-btn up" data-move="U">▲<small>위</small></button>
            <button type="button" class="dir-btn left" data-move="L">◀<small>왼쪽</small></button>
            <button type="button" class="dir-btn down" data-move="D">▼<small>아래</small></button>
            <button type="button" class="dir-btn right" data-move="R">▶<small>오른쪽</small></button>
          </div>`;
      } else if (type === 'arithmetic') {
        const q = payload.question.replace('?', `<span class="blank">${state.choice === '' ? '?' : payload.options[Number(state.choice)]}</span>`);
        els.stageBody.innerHTML = `<div class="question">${q}</div>
          <div class="options">${payload.options.map((opt, i) =>
            `<button type="button" class="option-btn ${state.choice === String(i) ? 'selected' : ''}" data-idx="${i}">${opt}</button>`
          ).join('')}</div>`;
      } else if (type === 'initials_quiz') {
        els.stageBody.innerHTML = `<div class="tiles" aria-label="초성 ${payload.clue}">${Array.from(payload.clue).map(ch =>
            `<span class="tile">${ch}</span>`).join('')}</div>
          <div class="options">${payload.options.map((opt, i) =>
            `<button type="button" class="option-btn ${state.choice === String(i) ? 'selected' : ''}" data-idx="${i}">${opt}</button>`
          ).join('')}</div>`;
      }
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
      const usedHeight = shell.offsetHeight;
      grid.style.display = '';
      grid.style.gridTemplateColumns = '';
      const gap = parseFloat(getComputedStyle(grid).gap) || 14;
      const count = grid.children.length;
      const availHeight = window.innerHeight - usedHeight;
      const availWidth = grid.clientWidth;
      let cols = 1;
      let size = 0;
      for (let c = 1; c <= count; c++) {
        const rows = Math.ceil(count / c);
        const fit = Math.min((availWidth - gap * (c - 1)) / c, (availHeight - gap * (rows - 1)) / rows);
        if (fit > size) { size = fit; cols = c; }
      }
      const cardSize = Math.max(56, Math.floor(Math.min(size, 200)));
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
    window.addEventListener('resize', fitToViewport);

    async function api(path, method = 'GET', body = null) {
      const options = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) options.body = JSON.stringify(body);
      const res = await fetch(`${API_ROOT}${path}`, options);
      let json;
      try {
        json = await res.json();
      } catch {
        // nginx 오류 페이지(HTML) 등 JSON이 아닌 응답: 서버 재시작 중이거나 연결 실패
        throw new Error(`잠시 연결이 어려워요 (${res.status}). 조금 뒤에 다시 눌러 주세요.`);
      }
      if (!res.ok || !json.success) throw new Error(json.message || '잠시 문제가 생겼어요. 다시 눌러 주세요.');
      return json.data;
    }

    function resetLocal() {
      state.memory = { selected: [], matched: new Set() };
      state.choice = '';
      if (state.stage?.stageType === 'maze') {
        state.maze = { path: '', x: state.stage.payload.start.x, y: state.stage.payload.start.y };
      }
    }

    async function load() {
      try {
        const data = await api(`/api/v1/games/${BOOT.gameSlug}/state?userId=${BOOT.userId}`);
        state.progress = data.progress;
        state.stage = data.stage;
        resetLocal();
        renderUi();
      } catch (e) { showFeedback(e.message, 'try'); }
    }

    els.stageBody.onclick = (e) => {
      const btn = e.target.closest('button');
      if (!btn || state.busy || state.progress?.cleared || !state.stage) return;

      if (btn.dataset.id) {
        const { id, key } = btn.dataset;
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
            if (stageReady()) { showFeedback('모두 찾으셨어요! 다음 문제로 넘어가요.', 'ok'); setTimeout(submit, 1200); }
            else { showFeedback('짝을 찾으셨어요!', 'ok'); chime('ok'); }
          } else {
            showFeedback('괜찮아요, 위치를 기억해 두고 다시 찾아봐요.', 'info');
            setTimeout(() => { state.memory.selected = []; renderStageContent(); }, 1300);
          }
        }
      } else if (btn.dataset.move) {
        const mv = btn.dataset.move;
        const grid = state.stage.payload.grid;
        let nx = state.maze.x, ny = state.maze.y;
        if (mv === 'U') ny--; else if (mv === 'D') ny++; else if (mv === 'L') nx--; else if (mv === 'R') nx++;
        if (ny >= 0 && ny < grid.length && nx >= 0 && nx < grid[0].length && grid[ny][nx] !== '#') {
          state.maze.x = nx; state.maze.y = ny; state.maze.path += mv;
          if (grid[ny][nx] === 'G') { showFeedback('출구에 도착하셨어요!', 'ok'); setTimeout(submit, 1000); }
        } else {
          showFeedback('그쪽은 벽이에요. 다른 길로 가 볼까요?', 'info');
        }
      } else if (btn.dataset.idx !== undefined) {
        state.choice = btn.dataset.idx;
      }
      renderStageContent();
    };

    async function submit() {
      if (state.busy || !state.stage) return;
      const type = state.stage.stageType;
      let answer = null;
      if (type === 'memory_match') answer = { matchedPairs: Array.from(state.memory.matched) };
      else if (type === 'maze') answer = { path: state.maze.path };
      else if (state.choice !== '') answer = { value: state.choice };
      if (!answer) return;
      state.busy = true;
      els.submitBtn.disabled = true;
      try {
        const data = await api(`/api/v1/games/${BOOT.gameSlug}/answer`, 'POST', {
          userId: BOOT.userId, gameSlug: BOOT.gameSlug, stageNo: state.stage.stageNo, answer
        });
        chime(data.correct ? 'ok' : 'try');
        state.progress = data.progress;
        state.stage = data.stage;
        resetLocal();
      } catch (e) { showFeedback(e.message, 'try'); }
      state.busy = false;
      if (state.stage || state.progress?.cleared) renderUi();
    }

    els.submitBtn.onclick = submit;
    els.resetBtn.onclick = async () => {
      if (!confirm('1번 문제부터 다시 시작할까요? (지금까지 모은 점수는 그대로 남아요)')) return;
      try {
        const data = await api(`/api/v1/games/${BOOT.gameSlug}/reset`, 'POST', { userId: BOOT.userId, gameSlug: BOOT.gameSlug });
        state.progress = data.progress;
        state.stage = data.stage;
        resetLocal();
        renderUi();
      } catch (e) { showFeedback(e.message, 'try'); }
    };

    renderSoundBtn();
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
