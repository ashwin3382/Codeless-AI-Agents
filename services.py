import os
import json
import redis as sync_redis
from dotenv import load_dotenv

# Core LangChain Object Imports
from langchain_core.documents import Document

# Third-Party Framework Connectors
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_milvus import Milvus
from langchain_community.chat_message_histories import RedisChatMessageHistory
from pymilvus import connections
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
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

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

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name="document_rag_collection",
    auto_id=True
)

memory_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name="chat_history_collection",
    auto_id=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ==========================================
# 4. RAG RETRIEVAL (DYNAMIC & AGENT-DRIVEN)
# ==========================================
def retrieve_relevant_docs(question: str) -> str:
    """
    Fetches matching context chunks from Milvus and returns them formatted.
    User-created agents read this output and apply their own configured system prompt.
    """
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant documents found."

    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        formatted_chunks.append(f"[Chunk {i} | Source: {source}]\n{doc.page_content}")

    return "\n\n".join(formatted_chunks)


# ==========================================
# 5. CORE WORKSPACE FUNCTIONS
# ==========================================
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        ttl=int(os.getenv("REDIS_HISTORY_TTL_SECONDS", "3600"))
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
        docs = memory_store.similarity_search(
            query=query or session_id,
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


def trigger_ingestion_task(bucket_name: str, object_name: str, username: str = None) -> str:
    from tasks import process_minio_document_bg

    clean_bucket = bucket_name.strip("'\"")
    clean_object = object_name.strip("'\"")

    task = process_minio_document_bg.delay(clean_bucket, clean_object, username)
    return f"Ingestion task started successfully! Task ID: {task.id}."