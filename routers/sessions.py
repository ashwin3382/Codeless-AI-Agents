import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from dependencies import get_db, get_current_user
from models import AgentSessionModel, AgentChatModel, AgentModel
from schemas import AgentSessionSchema, AgentChatSchema, AgentSessionCreateSchema

router = APIRouter(prefix="/sessions", tags=["agent-sessions"])


def _get_owned_session(db: Session, session_id: str, current_user: str) -> AgentSessionModel:
    session = db.query(AgentSessionModel).filter(AgentSessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Agent session not found.")
    if session.username != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to access this session.")
    return session


@router.post("", response_model=AgentSessionSchema, status_code=status.HTTP_201_CREATED)
def create_session(payload: AgentSessionCreateSchema, db: Session = Depends(get_db),
                   current_user: str = Depends(get_current_user)):
    if not db.query(AgentModel).filter(AgentModel.agent_name == payload.agent_name).first():
        raise HTTPException(status_code=404, detail="Agent structure configuration not found.")

    session = AgentSessionModel(
        id=uuid.uuid4().hex,
        agent_name=payload.agent_name,
        username=current_user,
        title=payload.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("", response_model=List[AgentSessionSchema])
def list_sessions(agent_name: Optional[str] = None, db: Session = Depends(get_db),
                  current_user: str = Depends(get_current_user)):
    q = db.query(AgentSessionModel).filter(AgentSessionModel.username == current_user)
    if agent_name:
        q = q.filter(AgentSessionModel.agent_name == agent_name)
    return q.order_by(AgentSessionModel.updated_at.desc()).all()


@router.get("/{session_id}", response_model=AgentSessionSchema)
def get_session(session_id: str, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    return _get_owned_session(db, session_id, current_user)


@router.get("/{session_id}/chats", response_model=List[AgentChatSchema])
def list_session_chats(session_id: str, db: Session = Depends(get_db),
                       current_user: str = Depends(get_current_user)):
    _get_owned_session(db, session_id, current_user)
    return (
        db.query(AgentChatModel)
        .filter(AgentChatModel.session_id == session_id)
        .order_by(AgentChatModel.created_at.asc())
        .all()
    )


@router.delete("/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    session = _get_owned_session(db, session_id, current_user)
    db.delete(session)
    db.commit()
    return {"status": "deleted", "session_id": session_id}