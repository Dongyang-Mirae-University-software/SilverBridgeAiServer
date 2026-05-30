from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.core.response import error_response, success_response
from app.database.session import get_db
from app.schemas.game_schema import GameAnswerRequest, GameStartRequest
from app.services.game_service import GameService

router = APIRouter(prefix="/api/v1/games", tags=["Games"])


@router.get("", summary="게임 목록 조회")
def list_games(db: Session = Depends(get_db)) -> dict:
    service = GameService(db)
    return success_response("게임 목록 조회 완료", service.list_games())


@router.post("/start", summary="게임 시작")
def start_game(payload: GameStartRequest, db: Session = Depends(get_db)) -> dict:
    service = GameService(db)
    try:
        result = service.start_game(payload.userId, payload.gameSlug)
    except HTTPException:
        raise
    return success_response("게임 시작 완료", result)


@router.get("/embed", summary="게임 iframe 페이지", response_class=HTMLResponse)
def game_embed(userId: int, gameSlug: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    service = GameService(db)
    return service.render_embed_html(request, userId, gameSlug)


@router.get("/{game_slug}/state", summary="게임 현재 상태 조회")
def get_game_state(game_slug: str, userId: int, db: Session = Depends(get_db)) -> dict:
    service = GameService(db)
    return success_response("게임 상태 조회 완료", service.get_state_payload(userId, game_slug))


@router.get("/{game_slug}/progress", summary="게임 진행 정보 조회")
def get_game_progress(game_slug: str, userId: int, db: Session = Depends(get_db)) -> dict:
    service = GameService(db)
    state = service.get_state_payload(userId, game_slug)
    return success_response("게임 진행 정보 조회 완료", state["progress"])


@router.get("/progress", summary="사용자별 게임 진행 목록 조회")
def list_user_progress(userId: int, db: Session = Depends(get_db)) -> dict:
    service = GameService(db)
    return success_response("사용자별 게임 진행 목록 조회 완료", service.list_progress_for_user(userId))


@router.post("/{game_slug}/answer", summary="게임 정답 제출")
def submit_game_answer(game_slug: str, payload: GameAnswerRequest, db: Session = Depends(get_db)) -> dict:
    service = GameService(db)
    try:
        result = service.submit_answer(payload.userId, game_slug, payload.stageNo, payload.answer)
    except HTTPException:
        raise
    return success_response("게임 정답 제출 완료", result)


@router.post("/{game_slug}/reset", summary="게임 초기화")
def reset_game(game_slug: str, payload: GameStartRequest, db: Session = Depends(get_db)) -> dict:
    service = GameService(db)
    try:
        result = service.reset_game(payload.userId, game_slug)
    except HTTPException:
        raise
    return success_response("게임 초기화 완료", result)
