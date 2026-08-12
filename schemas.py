from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MCPServerSchema(BaseModel):
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None

class AgentSchema(BaseModel):
    agent_name: str
    description: Optional[str] = None
    agent_prompt: str
    guardrails: Optional[str] = None
    tools: List[str] = Field(..., description="List matching elements: ['server_id:tool_name']")

class ChatPayload(BaseModel):
    # Optional now: omit it to start a brand new Agent Session (the server
    # generates the id and hands it back on the response / SSE init event).
    # Pass back a previously-issued session_id to continue that session.
    session_id: Optional[str] = None
    message: str

class ToolExecutionTrace(BaseModel):
    tool_name: str
    input_arguments: Dict[str, Any]
    output_data: str

class TelemetryReport(BaseModel):
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_token_usage: int = 0
    total_estimated_cost_usd: float = 0.0
    tool_calls_executed: List[ToolExecutionTrace] = []

class SyncChatResponse(BaseModel):
    session_id: str
    response: str
    telemetry: TelemetryReport

class UserCreateSchema(BaseModel):
    username: str
    password: str


# ==========================================
# Agent Sessions / Agent Chats
# ==========================================
class AgentSessionCreateSchema(BaseModel):
    agent_name: str
    title: Optional[str] = None

class AgentSessionSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    agent_name: str
    username: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class AgentChatSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    session_id: str
    role: str
    content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    created_at: datetime


# ==========================================
# Notifications
# ==========================================
class NotificationSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    type: str
    title: str
    message: Optional[str] = None
    session_id: Optional[str] = None
    is_read: bool
    created_at: datetime
