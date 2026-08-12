from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from dependencies import get_db, get_current_user
from models import AgentModel
from schemas import AgentSchema

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("")
def list_agents(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    return db.query(AgentModel).all()


@router.post("", status_code=status.HTTP_201_CREATED)
def create_agent(payload: AgentSchema, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    if db.query(AgentModel).filter(AgentModel.agent_name == payload.agent_name).first():
        raise HTTPException(status_code=400, detail="Agent structure configuration already exists.")

    agent = AgentModel(
        agent_name=payload.agent_name,
        description=payload.description,
        agent_prompt=payload.agent_prompt,
        guardrails=payload.guardrails,
        tools=payload.tools
    )
    db.add(agent)
    db.commit()
    return {"status": "created", "agent_name": payload.agent_name}


@router.put("/{agent_name}")
def update_agent(agent_name: str, payload: AgentSchema, db: Session = Depends(get_db),
                 current_user: str = Depends(get_current_user)):
    agent = db.query(AgentModel).filter(AgentModel.agent_name == agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Missing Agent profile context.")

    agent.description = payload.description
    agent.agent_prompt = payload.agent_prompt
    agent.guardrails = payload.guardrails
    agent.tools = payload.tools
    db.commit()
    return {"status": "updated", "agent_name": agent_name}


@router.delete("/{agent_name}")
def delete_agent(agent_name: str, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    agent = db.query(AgentModel).filter(AgentModel.agent_name == agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Missing target configuration context.")

    db.delete(agent)
    db.commit()
    return {"status": "deleted", "agent_name": agent_name}