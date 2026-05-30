from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class LiveWebSocketManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._subscriptions: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.broadcast(payload, session_id=session_id))


live_ws_manager = LiveWebSocketManager()
