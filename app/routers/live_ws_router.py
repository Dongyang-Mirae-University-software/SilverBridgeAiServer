from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.database.session import SessionLocal
from app.services.live_ws_manager import live_ws_manager
from app.services.stream_session_service import StreamSessionService, frame_store

router = APIRouter(tags=["LiveSocket"])


@router.websocket("/api/v1/ws/live")
async def live_websocket(websocket: WebSocket) -> None:
    settings = get_settings()
    token = websocket.headers.get("x-api-key") or websocket.query_params.get("apiKey")
    if token != settings.api_key:
        await websocket.close(code=1008, reason="AUTH_INVALID_KEY")
        return

    await live_ws_manager.connect(websocket)
    await websocket.send_json({"type": "connected", "data": {"message": "live socket connected"}})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "data": {"message": "invalid json"}})
                continue

            action = payload.get("action")
            session_id = payload.get("sessionId")

            if action == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if action == "list":
                db = SessionLocal()
                try:
                    service = StreamSessionService(db, frame_store)
                    items = service.list_live()
                finally:
                    db.close()
                await websocket.send_json({"type": "live_streams", "data": items})
                continue

            if action == "subscribe" and isinstance(session_id, str) and session_id:
                await live_ws_manager.subscribe(websocket, session_id)
                db = SessionLocal()
                try:
                    service = StreamSessionService(db, frame_store)
                    session = service.require_session(session_id)
                    status_payload = service.get_status_payload(session_id)
                    latest_analysis = service.latest_analysis_for_session(session_id, session.camera_identifier)
                except Exception:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "data": {"message": "session not found", "sessionId": session_id},
                        },
                    )
                    continue
                finally:
                    db.close()

                await websocket.send_json({"type": "subscribed", "sessionId": session_id})
                await websocket.send_json(
                    {"type": "session_status", "sessionId": session_id, "data": status_payload},
                )
                await websocket.send_json(
                    {
                        "type": "latest_analysis",
                        "sessionId": session_id,
                        "data": latest_analysis,
                    },
                )
                continue

            if action == "unsubscribe" and isinstance(session_id, str) and session_id:
                await live_ws_manager.unsubscribe(websocket, session_id)
                await websocket.send_json({"type": "unsubscribed", "sessionId": session_id})
                continue

            await websocket.send_json({"type": "error", "data": {"message": "unsupported action"}})

    except WebSocketDisconnect:
        await live_ws_manager.disconnect(websocket)
