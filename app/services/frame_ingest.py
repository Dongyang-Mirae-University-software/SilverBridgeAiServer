"""프레임 한 장의 수신 처리 - JPEG POST(/frame)와 H.264 WS(ingest-ws)가 같이 쓴다.

저장 -> 분석 -> 세션 상태·최신 분석 방송 묶음을 한 곳에 둔다. 두 경로가 따로 복제하면 한쪽만 고쳐져 어긋난다.
"""

from __future__ import annotations

from app.services.live_ws_manager import live_ws_manager
from app.services.stream_session_service import StreamSessionService, StreamSessionState


async def process_ingested_frame(
    service: StreamSessionService,
    session: StreamSessionState,
    frame_bytes: bytes,
) -> StreamSessionState:
    updated = service.ingest_frame(session, frame_bytes)
    latest_analysis = await service.analyze_stream_frame_async(updated.session_id, frame_bytes)
    status_payload = service.get_status_payload(updated.session_id)
    live_ws_manager.broadcast_nowait(
        {"type": "session_status", "sessionId": updated.session_id, "data": status_payload},
        session_id=updated.session_id,
    )
    live_ws_manager.broadcast_nowait(
        {"type": "latest_analysis", "sessionId": updated.session_id, "data": latest_analysis},
        session_id=updated.session_id,
    )
    return updated
