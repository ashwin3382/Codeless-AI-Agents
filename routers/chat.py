import json
import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from dependencies import get_db, get_current_user, redis_client
from models import AgentModel
from schemas import ChatPayload, SyncChatResponse
from services import SessionLocal, notify_user
from agent_engine import run_agent_workflow

router = APIRouter(prefix="/agents", tags=["chat"])


@router.post("/{agent_name}/chat")
async def chat_with_agent_stream(agent_name: str, payload: ChatPayload, db: Session = Depends(get_db),
                                 current_user: str = Depends(get_current_user)):
    agent_cfg = db.query(AgentModel).filter(AgentModel.agent_name == agent_name).first()
    if not agent_cfg:
        raise HTTPException(status_code=404, detail="Agent structure configuration not found.")

    if not payload.session_id:
        payload.session_id = uuid.uuid4().hex

    agent_name_snapshot = agent_cfg.agent_name
    current_user_snapshot = current_user
    pubsub_channel = f"chat:{payload.session_id}"

    async def process_runtime_stream():
        bg_db = SessionLocal()
        try:
            bg_agent_cfg = bg_db.query(AgentModel).filter(AgentModel.agent_name == agent_name_snapshot).first()
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
            notify_user(bg_db, current_user_snapshot, "Agent chat failed", message=str(e), notif_type="error",
                        session_id=payload.session_id)
        finally:
            bg_db.close()

    asyncio.create_task(process_runtime_stream())

    async def sse_pubsub_listener():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(pubsub_channel)
        yield f"data: {json.dumps({'event': 'init', 'channel': pubsub_channel, 'session_id': payload.session_id})}\n\n"

        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if message:
                    data = json.loads(message['data'].decode('utf-8'))
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get('event') in ['done', 'error']:
                        break
                await asyncio.sleep(0.01)
        finally:
            await pubsub.unsubscribe(pubsub_channel)

    return StreamingResponse(sse_pubsub_listener(), media_type="text/event-stream")


@router.post("/{agent_name}/chat/json", response_model=SyncChatResponse)
async def chat_with_agent_swagger_fallback(agent_name: str, payload: ChatPayload, db: Session = Depends(get_db),
                                           current_user: str = Depends(get_current_user)):
    agent_cfg = db.query(AgentModel).filter(AgentModel.agent_name == agent_name).first()
    if not agent_cfg:
        raise HTTPException(status_code=404, detail="Agent structure configuration not found.")

    if not payload.session_id:
        payload.session_id = uuid.uuid4().hex

    try:
        response_text, telemetry_data = await run_agent_workflow(
            agent_cfg=agent_cfg,
            payload=payload,
            db=db,
            current_user=current_user
        )
    except Exception as e:
        db.rollback()
        notify_user(db, current_user, "Agent chat failed", message=str(e), notif_type="error",
                    session_id=payload.session_id)
        raise
    return SyncChatResponse(session_id=payload.session_id, response=response_text, telemetry=telemetry_data)