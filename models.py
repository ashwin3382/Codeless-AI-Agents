from services import Base
from sqlalchemy import Column, String, Text, JSON, Integer, Boolean, DateTime, ForeignKey, Float, func
from sqlalchemy.orm import relationship


class UserModel(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    hashed_password = Column(String, nullable=False)


class MCPServerModel(Base):
    __tablename__ = "mcp_servers"
    server_id = Column(String, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    command = Column(String, nullable=False)
    args = Column(JSON, nullable=False)
    env = Column(JSON, nullable=True)


class AgentModel(Base):
    __tablename__ = "agents"
    agent_name = Column(String, primary_key=True, index=True)
    description = Column(Text, nullable=True)
    agent_prompt = Column(Text, nullable=False)
    guardrails = Column(Text, nullable=True)
    tools = Column(JSON, nullable=False)  # Ex: ["server_id:tool_name"]


class AgentSessionModel(Base):
    """
    A single conversation thread between one user and one agent. This is the
    durable, queryable record of "a chat with agent X" - Redis (short-term,
    TTL'd) and Milvus (long-term semantic recall) remain the LLM-context
    stores used by agent_engine.run_agent_workflow; this table (plus
    AgentChatModel below) is the source of truth for session/chat history
    the API and UI list, page through, and let users delete.
    """
    __tablename__ = "agent_sessions"

    id = Column(String, primary_key=True, index=True)
    agent_name = Column(String, ForeignKey("agents.agent_name", ondelete="CASCADE"), nullable=False, index=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    chats = relationship(
        "AgentChatModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AgentChatModel.created_at",
    )


class AgentChatModel(Base):
    """One turn (human or ai message) belonging to an AgentSessionModel.
    A session has N chats - this is where the per-turn transcript and its
    token/cost telemetry live."""
    __tablename__ = "agent_chats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "human" | "ai"
    content = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    estimated_cost_usd = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("AgentSessionModel", back_populates="chats")


class NotificationModel(Base):
    """A single notification for a user - e.g. a background ingestion job
    finishing/failing, or an agent chat erroring out. Delivered two ways:
    polled via the /notifications REST endpoints, and pushed live over the
    /notifications/stream SSE endpoint (see services.notify_user)."""
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String, nullable=False, default="info")  # e.g. "info" | "success" | "error"
    title = Column(String, nullable=False)
    message = Column(Text, nullable=True)
    session_id = Column(String, nullable=True)  # optional link back to an AgentSessionModel or ingestion job
    is_read = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
