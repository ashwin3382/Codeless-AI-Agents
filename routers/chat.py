import json
import asyncio
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from dependencies import redis_client
from models import AgentModel
from schemas import ChatPayload
from services import SessionLocal, notify_user, SECRET_KEY, ALGORITHM
from agent_engine import run_agent_workflow

router = APIRouter(prefix="/agents", tags=["chat"])


def decode_ws_token(token: str) -> str:
    """
    Mirrors dependencies.get_current_user, adapted for WebSocket use.
    get_current_user itself can't be reused directly via Depends() here:
    it's built on OAuth2PasswordBearer, which expects a normal HTTP
    Request to pull the Authorization header from, and browsers can't set
    custom headers on a WebSocket handshake anyway - so the token arrives
    as a query param instead and is decoded the same way, manually.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise ValueError("Could not validate signature credentials.") from e
    username = payload.get("sub")
    if not username:
        raise ValueError("Invalid token details.")
    return username


@router.websocket("/{agent_name}/ws")
async def chat_with_agent_ws(websocket: WebSocket, agent_name: str):
    """
    Single persistent WebSocket connection for an agent chat session.
    Replaces the old SSE (`POST /{agent_name}/chat`) and JSON fallback
    (`POST /{agent_name}/chat/json`) endpoints entirely.

    Auth: browsers cannot set custom headers on a WebSocket handshake, so
    the bearer token is passed as a query param instead:
        wss://host/agents/{agent_name}/ws?token=<jwt>
    A non-browser client may alternatively send an Authorization header;
    both are checked.

    Client -> server messages (one per turn), JSON:
        {"message": "...", "session_id": "..." }   # session_id optional
    Server -> client messages, JSON, forwarded from the same pubsub
    events the old SSE stream used:
        {"event": "init", "channel": ..., "session_id": ...}
        {"event": "token", "text": "..."}           # streamed chunks
        {"event": "tool_start", "tool": "..."}
        {"event": "done"}
        {"event": "error", "detail": "..."}
    """
    await websocket.accept()

    token = websocket.query_params.get("token")
    if not token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()

    if not token:
        await websocket.close(code=4401, reason="Missing auth token")
        return

    try:
        current_user = decode_ws_token(token)
    except Exception:
        await websocket.close(code=4401, reason="Invalid or expired token")
        return

    db = SessionLocal()
    try:
        agent_cfg = db.query(AgentModel).filter(AgentModel.agent_name == agent_name).first()
        if not agent_cfg:
            await websocket.close(code=4404, reason="Agent structure configuration not found.")
            return

        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                data = json.loads(raw)
                payload = ChatPayload(**data)
            except Exception as e:
                await websocket.send_text(json.dumps({"event": "error", "detail": f"Invalid payload: {e}"}))
                continue

            if not payload.session_id:
                payload.session_id = uuid.uuid4().hex

            pubsub_channel = f"chat:{payload.session_id}"
            agent_name_snapshot = agent_cfg.agent_name
            current_user_snapshot = current_user

            async def process_runtime_stream():
                bg_db = SessionLocal()
                try:
                    bg_agent_cfg = bg_db.query(AgentModel).filter(
                        AgentModel.agent_name == agent_name_snapshot).first()
                    await run_agent_workflow(
                        agent_cfg=bg_agent_cfg,
                        payload=payload,
                        db=bg_db,
                        current_user=current_user_snapshot,
                        redis_client=redis_client,
                        pubsub_channel=pubsub_channel
                    )
                    await redis_client.publish(pubsub_channel, json.dumps({'event': 'done'}))
                except Exception as e:
                    bg_db.rollback()
                    await redis_client.publish(pubsub_channel, json.dumps({'event': 'error', 'detail': str(e)}))
                    notify_user(bg_db, current_user_snapshot, "Agent chat failed", message=str(e),
                                notif_type="error", session_id=payload.session_id)
                finally:
                    bg_db.close()

            asyncio.create_task(process_runtime_stream())

            pubsub = redis_client.pubsub()
            await pubsub.subscribe(pubsub_channel)
            await websocket.send_text(json.dumps(
                {'event': 'init', 'channel': pubsub_channel, 'session_id': payload.session_id}))

            try:
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                    if message:
                        evt = json.loads(message['data'].decode('utf-8'))
                        await websocket.send_text(json.dumps(evt))
                        if evt.get('event') in ('done', 'error'):
                            break
                    await asyncio.sleep(0.01)
            finally:
                await pubsub.unsubscribe(pubsub_channel)
            # loop back to receive_text() for the next turn on the same connection
    finally:
        db.close()

