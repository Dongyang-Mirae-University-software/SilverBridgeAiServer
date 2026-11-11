from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import MemberId


class GuardianWard(BaseModel):
    """보호자의 ACTIVE 연결 피보호자 한 명(BE 가 DB 값으로 채운다 - 2026-11-11). 예약 대상 선택지에만 쓴다.
    이름·관계는 선택지 표시 글자로만 쓰고(길이는 chat_service 가 자른다) 로그·MedGemma·userContext 에 넣지 않는다."""

    wardId: MemberId
    name: str | None = None
    relation: str | None = None


class ChatContext(BaseModel):
    age: int | None = None
    email: str | None = None
    name: str | None = None
    phone: str | None = None
    gender: str | None = None
    birthDate: str | None = None
    postcode: str | None = None
    address: str | None = None
    addressDetail: str | None = None
    guardianId: int | None = None
    location: str | None = None
    role: str | None = None
    # 보호자가 피보호자 대신 상담·예약할 때. 예약은 이 사람 계정으로 들어가고 환자명·연락처도 이 사람 것.
    wardId: str | None = None
    wardName: str | None = None
    wardPhone: str | None = None
    # 보호자 경로에서 BE 가 넣는 ACTIVE 연결 피보호자 목록(2026-11-11). None = 칸 없음(옛 BE - 예약 대상을 묻지 않고 이전 동작),
    # [] = 연결된 피보호자 없음(챗봇 예약 불가 안내). 보호자 챗봇 예약은 피보호자만 대상이다(보호자 웹 예약 화면과 같다).
    guardianWards: list[GuardianWard] | None = None


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class UiOption(BaseModel):
    value: str
    label: str


class UiPayload(BaseModel):
    kind: Literal["select", "date", "text"]
    field: str
    label: str
    options: list[UiOption] = Field(default_factory=list)
    placeholder: str | None = None


class UiSelection(BaseModel):
    field: str
    value: str


class ChatRequest(BaseModel):
    userId: MemberId
    message: str = ""
    sessionId: str | None = None
    history: list[ChatHistoryItem] = Field(default_factory=list)
    context: ChatContext | None = None
    uiSelection: UiSelection | None = None


class ChatReplyData(BaseModel):
    sessionId: str
    reply: str
    riskLevel: str
    recommendedAction: str
    reservationRequired: bool
    intent: str
    engine: str
    modelName: str | None = None
    type: str = "message"
    ui: UiPayload | None = None
    tool: str | None = None
    data: Any | None = None
    # 이어서 물어볼 질문 0~3개(2026-10-30), 되묻기 턴은 답 칩 4개(2026-11-02). 응급·자해·예약 선택지 답에는 빈 목록.
    suggestedQuestions: list[str] = Field(default_factory=list)
    # 칩 종류(2026-11-02): "answer"면 suggestedQuestions 는 되묻기에 눌러서 답하는 고정 칩 4개. 그 밖에는 칸 없음.
    suggestedKind: str | None = None
    # 답 모양(2026-11-01): headline·body·questions·homeCare·visitHospitalIf. 예약·선택지 답은 null - 단 위기 뒤 해제 턴(meta.crisisFollowup)이면 작은 layout(urgent=true, 109 줄만 headline)이 붙는다(#200). FE 는 있으면 이것으로 그린다.
    layout: dict[str, Any] | None = None


class ChatLogOut(BaseModel):
    id: int
    chatNo: str
    userId: str
    message: str
    contextJson: str
    reply: str
    riskLevel: str
    recommendedAction: str
    reservationRequired: bool
    createdAt: datetime
