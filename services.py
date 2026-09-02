import os
import json
import time
import redis as sync_redis
from dotenv import load_dotenv

# Core LangChain Object Imports
from langchain_core.documents import Document

# Third-Party Framework Connectors
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_milvus import Milvus
from langchain_community.chat_message_histories import RedisChatMessageHistory
from pymilvus import connections, MilvusException
from minio import Minio

# Production Database Layer & Security (Postgres, JWT, OWASP)
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

load_dotenv()

# ==========================================
# 1. ENTERPRISE POSTGRES & MINIO CONNECTIONS
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agent_db")
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

# How long a session may sit inactive before the Celery Beat sweep
# (tasks.expire_stale_sessions) deletes its Postgres row, its Milvus
# vectors, and its MinIO objects. 3 days while testing locally - bump this
# via env for anything closer to production.
SESSION_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "3"))


def _connect_milvus_with_retry(host: str, port: str, retries: int = 15, delay: float = 5.0) -> None:
    """Milvus standalone can still be finishing internal init even after its
    container/healthcheck reports up, so retry instead of hard-crashing the
    whole process on the first failed attempt (which just triggers a full
    container restart-loop under `restart: always`)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            connections.connect(alias="default", host=host, port=port)
            print(f"[services] Connected to Milvus at {host}:{port} (attempt {attempt}/{retries})")
            return
        except MilvusException as e:
            last_err = e
            print(f"[services] Milvus not ready yet (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Milvus at {host}:{port} after {retries} attempts") from last_err


_connect_milvus_with_retry(MILVUS_HOST, MILVUS_PORT)

# ==========================================
# 2. JWT & SECURITY CONTROLS (OWASP CONFORMANT)
# ==========================================
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = "dev-only-insecure-secret-do-not-use-in-production"
    print(
        "[services] WARNING: JWT_SECRET_KEY is not set. Using an insecure "
        "development default so the app can still run locally. Set "
        "JWT_SECRET_KEY before deploying anywhere real."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def seed_default_admin():
    from models import UserModel

    db = SessionLocal()
    try:
        if db.query(UserModel).count() == 0:
            username = os.getenv("ADMIN_USERNAME", "admin")
            password = os.getenv("ADMIN_PASSWORD", "changeme123")
            db.add(UserModel(username=username, hashed_password=hash_password(password)))
            db.commit()
            print(f"[services] Seeded default test user '{username}'.")
    finally:
        db.close()

def seed_default_mcp_server():
    """Auto-registers the RAG MCP server (a long-running SSE service now,
    not a subprocess) if it isn't already there, so a fresh DB doesn't
    require manually POSTing /mcp with the exact right URL every time."""
    from models import MCPServerModel

    db = SessionLocal()
    try:
        if db.query(MCPServerModel).count() == 0:
            mcp_url = os.getenv("MCP_SERVICE_URL", "http://mcp_service:8000/sse")
            db.add(MCPServerModel(
                server_id="rag_manager",
                name="RAG Manager",
                command=mcp_url,
                args=[],
                env=None,
            ))
            db.commit()
            print(f"[services] Seeded default MCP server 'rag_manager' -> {mcp_url}")
    finally:
        db.close()

# ==========================================
# 2b. NOTIFICATIONS
# ==========================================
_notif_redis = sync_redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))


def notify_user(db, username: str, title: str, message: str = None, notif_type: str = "info",
                session_id: str = None):
    if not username:
        return None

    from models import NotificationModel

    notif = NotificationModel(username=username, type=notif_type, title=title, message=message,
                              session_id=session_id)
    db.add(notif)
    db.commit()
    db.refresh(notif)

    try:
        _notif_redis.publish(f"notifications:{username}", json.dumps({
            "event": "notification",
            "id": notif.id,
            "type": notif.type,
            "title": notif.title,
            "message": notif.message,
            "session_id": notif.session_id,
            "is_read": notif.is_read,
            "created_at": notif.created_at.isoformat() if notif.created_at else None,
        }))
    except Exception as e:
        print(f"[services] Failed to publish live notification for '{username}': {e}")

    return notif


# ==========================================
# 3. AI CORE DEPLOYMENTS (LANGCHAIN)
# ==========================================
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    streaming=True
)

def _init_milvus_store_with_retry(retries: int = 15, delay: float = 5.0, **kwargs) -> Milvus:
    """langchain_milvus opens its own gRPC/HTTP connection to Milvus separate
    from the pymilvus `connections.connect()` call above, so it needs its
    own retry guard against the same standalone-Milvus startup race."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            store = Milvus(**kwargs)
            print(f"[services] Initialized Milvus store '{kwargs.get('collection_name')}' "
                  f"(attempt {attempt}/{retries})")
            return store
        except Exception as e:
            last_err = e
            print(f"[services] Milvus store '{kwargs.get('collection_name')}' not ready "
                  f"(attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError(
        f"Could not initialize Milvus store '{kwargs.get('collection_name')}' "
        f"after {retries} attempts"
    ) from last_err


# NOTE: collection names bumped to _v2 because adding partition_key_field
# to an *existing* collection isn't a schema migration langchain-milvus
# does for you - it has to be set at collection-creation time. If you
# already have a "document_rag_collection" from before this change, its
# old (un-partitioned, cross-session) data lives there and is NOT migrated;
# it's simplest to just drop it and re-ingest documents per-session.
#
# partition_key_field="session_id" is Milvus's own recommended multi-tenant
# pattern: every document/chunk written here MUST carry "session_id" in its
# metadata (tasks.py and add_to_memory() both do this), Milvus hash-routes
# it into one of its internal partitions, and every query below still
# passes an explicit `expr` filter on session_id as a hard boundary -
# so a bug in the partition routing itself still can't leak across sessions.
vector_store = _init_milvus_store_with_retry(
    embedding_function=embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name="document_rag_collection_v2",
    auto_id=True,
    partition_key_field="session_id",
    index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}},
    search_params={"metric_type": "COSINE", "params": {"ef": 64}}
)

memory_store = _init_milvus_store_with_retry(
    embedding_function=embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name="chat_history_collection_v2",
    auto_id=True,
    partition_key_field="session_id",
    index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}},
    search_params={"metric_type": "COSINE", "params": {"ef": 64}}
)


# ==========================================
# 4. RAG RETRIEVAL (DYNAMIC & AGENT-DRIVEN, SESSION-SCOPED)
# ==========================================
def retrieve_relevant_docs(question: str, session_id: str) -> str:
    """
    Fetches matching context chunks from Milvus and returns them formatted
    for the calling agent's LLM. session_id is mandatory and always comes
    from a server-verified source (the MCP middleware's context state, not
    a tool argument the LLM could set) - see mcp_server.py.

    Deliberately avoids "[Chunk N]" framing and raw minio:// paths here -
    agents tend to echo whatever labels appear in tool output back to the
    end user verbatim, so keep this clean and add an explicit instruction
    not to surface it as a citation format.
    """
    if not session_id:
        return "No session context available for retrieval."

    # session_id is always a server-generated uuid4().hex, never raw user
    # input, so building the expr string this way is safe from injection.
    docs = vector_store.similarity_search(
        question,
        k=4,
        expr=f"session_id == '{session_id}'",
    )
    if not docs:
        return "No relevant documents found."

    formatted_chunks = []
    seen_sources = set()
    for doc in docs:
        raw_source = doc.metadata.get("source", "Unknown Source")
        # "minio://bucket/<session_id>/module 5 EVS notes.pdf" -> "module 5 EVS notes.pdf"
        display_source = raw_source.rsplit("/", 1)[-1] if raw_source != "Unknown Source" else raw_source
        seen_sources.add(display_source)
        formatted_chunks.append(f"(From: {display_source})\n{doc.page_content}")

    context = "\n\n---\n\n".join(formatted_chunks)
    note = (
        "[Internal retrieval context below - for answering the question only. "
        "Do not mention chunk numbers, internal formatting, or file paths in "
        "your reply. If asked where information came from, refer to it "
        f"naturally by document name only: {', '.join(sorted(seen_sources))}.]"
    )
    return f"{note}\n\n{context}"


# ==========================================
# 5. CORE WORKSPACE FUNCTIONS
# ==========================================
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        # Was defaulting to 1 hour - bumped default to match SESSION_TTL_DAYS
        # (3 days locally) so short-term chat memory doesn't expire well
        # before the session itself does.
        ttl=int(os.getenv("REDIS_HISTORY_TTL_SECONDS", str(SESSION_TTL_DAYS * 24 * 3600)))
    )


def add_to_memory(session_id: str, user_query: str, ai_response: str) -> str:
    user_doc = Document(
        page_content=user_query,
        metadata={"session_id": session_id, "role": "human"}
    )
    ai_doc = Document(
        page_content=ai_response,
        metadata={"session_id": session_id, "role": "ai"}
    )
    memory_store.add_documents([user_doc, ai_doc])
    return f"Successfully saved conversation turn for session: {session_id}"


def retrieve_from_memory(session_id: str, query: str = "", k: int = 6) -> list:
    from langchain_core.messages import HumanMessage, AIMessage
    try:
        # NOTE: this was previously called with `filter=...`, which isn't
        # the langchain-milvus similarity_search kwarg - the actual
        # partition/metadata filter parameter is `expr`. Fixed here; this
        # matters more now than before since it's the hard boundary that
        # keeps one session's long-term memory out of another's recall.
        search_target = query if query and query.strip() else "conversation history context"
        docs = memory_store.similarity_search(
            query=search_target,
            k=k,
            expr=f"session_id == '{session_id}'"
        )
        history = []
        for doc in reversed(docs):
            role = doc.metadata.get("role")
            if role == "human":
                history.append(HumanMessage(content=doc.page_content))
            elif role == "ai":
                history.append(AIMessage(content=doc.page_content))
        return history
    except Exception as e:
        print(f"Error fetching long-term memory or memory empty: {e}")
        return []


def purge_session_data(session_id: str, username: str = None) -> None:
    """Deletes all vector data (uploaded-document chunks + long-term chat
    memory) for one session. Called by tasks.expire_stale_sessions once a
    session's Postgres row has passed SESSION_TTL_DAYS of inactivity - safe
    to call even for a session that never had any vectors written."""
    expr = f"session_id == '{session_id}'"
    for store, label in ((vector_store, "document_rag_collection_v2"), (memory_store, "chat_history_collection_v2")):
        try:
            store.delete(expr=expr)
        except Exception as e:
            print(f"[services] Warning: could not purge '{label}' for session {session_id} "
                  f"(user={username}): {e}")


def trigger_ingestion_task(bucket_name: str, object_name: str, session_id: str, username: str = None) -> dict:
    from tasks import process_minio_document_bg

    clean_bucket = bucket_name.strip("'\"")
    clean_object = object_name.strip("'\"")

    storage_key = f"{session_id}/{clean_object}" if session_id else clean_object

    task = process_minio_document_bg.delay(clean_bucket, storage_key, session_id, username)
    return {
        "task_id": task.id,
        "storage_key": storage_key,
        "detail": f"Ingestion task started successfully! Task ID: {task.id}.",
    }