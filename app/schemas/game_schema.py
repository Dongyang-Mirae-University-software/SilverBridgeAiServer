from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GameStartRequest(BaseModel):
    userId: int = Field(gt=0)
    gameSlug: str = Field(min_length=1)


class GameAnswerRequest(BaseModel):
    userId: int = Field(gt=0)
    gameSlug: str = Field(min_length=1)
    stageNo: int | None = None
    answer: Any = None


class GameStateRequest(BaseModel):
    userId: int = Field(gt=0)
    gameSlug: str = Field(min_length=1)


class GameCatalogOut(BaseModel):
    slug: str
    title: str
    description: str
    totalStages: int
    themeColor: str


class GameStageOut(BaseModel):
    stageNo: int
    title: str
    stageType: str
    prompt: str
    payload: dict[str, Any]
    maxScore: int


class GameProgressOut(BaseModel):
    userId: int
    gameSlug: str
    currentStageNo: int
    score: int
    attempts: int
    cleared: bool
    lastAnswerCorrect: bool | None
    state: dict[str, Any]
    startedAt: str
    updatedAt: str
    clearedAt: str | None


class GameStateOut(BaseModel):
    game: GameCatalogOut
    progress: GameProgressOut
    stage: GameStageOut | None = None
    totalStages: int
    completed: bool
    iframeUrl: str


class GameSubmitOut(BaseModel):
    correct: bool
    message: str
    scoreDelta: int
    progress: GameProgressOut
    stage: GameStageOut | None = None
    completed: bool


class GameAttemptOut(BaseModel):
    stageNo: int
    attemptNo: int
    correct: bool
    scoreDelta: int
    createdAt: str
