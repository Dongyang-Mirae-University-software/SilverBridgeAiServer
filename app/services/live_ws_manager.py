from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

from app.core.config import get_settings

_LOG = logging.getLogger(__name__)


class LiveWebSocketManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._subscriptions: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
        # 동기 엔드포인트(스레드풀)에서 방송할 때 넘겨줄 이벤트 루프. 앱 시작(lifespan)과 첫 접속 때 잡아 둔다.
        self._loop: asyncio.AbstractEventLoop | None = None
        # 방송 작업 참조 - 쥐고 있지 않으면 실행 중에 GC 로 사라질 수 있다(A-5). 끝나면 스스로 뺀다.
        self._tasks: set[Any] = set()

    @staticmethod
    def _send_timeout() -> float:
        try:
            value = float(getattr(get_settings(), "live_ws_send_timeout_sec", 5.0))
        except Exception:  # noqa: BLE001
            value = 5.0
        return value if value > 0 else 5.0

    def _track(self, task: Any) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

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

        # 구독자마다 따로·동시에 보낸다 - 막힌 구독자 하나가 다른 구독자 전송을 밀지 않게. 시간 안에 못 보내면 구독에서 뺀다.
        timeout = self._send_timeout()
        outcomes = await asyncio.gather(
            *(self._send_one(ws, payload, timeout) for ws in targets), return_exceptions=True,
        )
        stale = [ws for ws, ok in zip(targets, outcomes) if ok is not True]
        for ws in stale:
            await self.disconnect(ws)
        if stale:
            _LOG.warning("[WS-SUBSCRIBER-DROPPED] count=%d type=%s", len(stale), payload.get("type"))

    @staticmethod
    async def _send_one(websocket: WebSocket, payload: dict[str, Any], timeout: float) -> bool:
        try:
            await asyncio.wait_for(websocket.send_json(payload), timeout=timeout)
            return True
        except Exception:  # noqa: BLE001 - 시간 초과·전송 실패 모두 구독 해제 대상
            return False

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
            self._track(running.create_task(self.broadcast(payload, session_id=session_id)))
            return

        loop = self._loop
        if loop is None or loop.is_closed():
            if self._connections:
                _LOG.warning("[WS-BROADCAST-DROPPED] 이벤트 루프 없음 - 방송 유실 type=%s", payload.get("type"))
            return  # 접속한 클라이언트가 없으면 보낼 곳도 없다
        try:
            self._track(asyncio.run_coroutine_threadsafe(self.broadcast(payload, session_id=session_id), loop))
        except RuntimeError:  # 종료 중 루프
            _LOG.warning("[WS-BROADCAST-DROPPED] 이벤트 루프 종료 중 - 방송 유실 type=%s", payload.get("type"))


live_ws_manager = LiveWebSocketManager()
