from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

_LOG = logging.getLogger(__name__)


class LiveWebSocketManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._subscriptions: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
        # 동기 엔드포인트(스레드풀)에서 방송할 때 넘겨줄 이벤트 루프. 앱 시작(lifespan)과 첫 접속 때 잡아 둔다.
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        self._loop = asyncio.get_running_loop()
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
            for session_id in list(self._subscriptions.keys()):
                subscribers = self._subscriptions[session_id]
                subscribers.discard(websocket)
                if not subscribers:
                    self._subscriptions.pop(session_id, None)

    async def subscribe(self, websocket: WebSocket, session_id: str) -> None:
        async with self._lock:
            self._subscriptions[session_id].add(websocket)

    async def unsubscribe(self, websocket: WebSocket, session_id: str) -> None:
        async with self._lock:
            subscribers = self._subscriptions.get(session_id)
            if not subscribers:
                return
            subscribers.discard(websocket)
            if not subscribers:
                self._subscriptions.pop(session_id, None)

    async def broadcast(self, payload: dict[str, Any], session_id: str | None = None) -> None:
        async with self._lock:
            if session_id:
                targets = list(self._subscriptions.get(session_id, set()))
            else:
                targets = list(self._connections)
        if not targets:
            return

        stale: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        for ws in stale:
            await self.disconnect(ws)

    def broadcast_nowait(self, payload: dict[str, Any], session_id: str | None = None) -> None:
        """기다리지 않고 방송한다. 이벤트 루프 안이든 스레드풀(동기 def 엔드포인트)이든 호출할 수 있다.

        예전에는 실행 중인 루프가 없으면 조용히 버려, 세션 생성·종료(동기 엔드포인트) 때의
        live_streams·session_status 방송이 전부 사라졌다(QA AI-1). 백엔드는 이 방송으로 새 세션을 구독한다.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            running.create_task(self.broadcast(payload, session_id=session_id))
            return

        loop = self._loop
        if loop is None or loop.is_closed():
            if self._connections:
                _LOG.warning("[WS-BROADCAST-DROPPED] 이벤트 루프 없음 - 방송 유실 type=%s", payload.get("type"))
            return  # 접속한 클라이언트가 없으면 보낼 곳도 없다
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(payload, session_id=session_id), loop)
        except RuntimeError:  # 종료 중 루프
            _LOG.warning("[WS-BROADCAST-DROPPED] 이벤트 루프 종료 중 - 방송 유실 type=%s", payload.get("type"))


live_ws_manager = LiveWebSocketManager()
