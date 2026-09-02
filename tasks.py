import os
import uuid
from celery import Celery
from services import vector_store, minio_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
)

celery_app = Celery(
    "ingest_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)


def load_bytes_via_tempfile(file_bytes: bytes, filename: str):
    # object_name comes from MinIO / user-supplied input - strip any path
    # components before touching the filesystem, or a name like
    # "../../etc/something" could escape /tmp.
    safe_filename = os.path.basename(filename)
    ext = os.path.splitext(safe_filename)[1].lower()

    loaders = {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader
    }

    if ext not in loaders:
        raise ValueError(f"Extension '{ext}' is unsupported.")

    # Unique temp path so two concurrent ingestion tasks for files that share
    # a basename can't overwrite each other's temp file mid-read.
    tmp_path = f"/tmp/{uuid.uuid4().hex}_{safe_filename}"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        docs = loaders[ext](tmp_path).load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return docs


def _notify_ingestion_result(username: str, title: str, message: str, notif_type: str, bucket_name: str,
                             object_name: str):
    """Best-effort notification for whoever kicked off this ingestion job.
    Runs in the Celery worker process, so it opens its own short-lived DB
    session rather than sharing one across tasks."""
    if not username:
        return
    from services import SessionLocal, notify_user
    db = SessionLocal()
    try:
        notify_user(db, username, title, message=message, notif_type=notif_type,
                    session_id=f"minio://{bucket_name}/{object_name}")
    finally:
        db.close()


@celery_app.task(bind=True)
def process_minio_document_bg(self, bucket_name: str, object_name: str, session_id: str = None,
                              username: str = None):
    """Downloads files from MinIO storage and handles chunk parsing in a background worker.

    session_id is required for anything going through services.trigger_ingestion_task
    (the documents.py /upload path) - it's what puts each chunk into the right
    Milvus partition (see vector_store's partition_key_field="session_id" in
    services.py) so one session's documents can never leak into another
    session's RAG retrieval. It's optional only because mcp_server.py's
    direct `ingest_minio_document` tool still calls this with just
    (bucket_name, object_name) for ad-hoc/manual ingestion outside a session."""
    try:
        self.update_state(state="PROGRESS", meta={"msg": "Downloading from MinIO..."})
        response = minio_client.get_object(bucket_name, object_name)
        file_bytes = response.read()
        response.close()
        response.release_conn()

        self.update_state(state="PROGRESS", meta={"msg": "Parsing document structure..."})
        raw_docs = load_bytes_via_tempfile(file_bytes, object_name)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(raw_docs)

        for chunk in chunks:
            # Keep the loader's own metadata (e.g. page numbers) - only add
            # to it, don't wipe everything else with .clear().
            chunk.metadata["source"] = f"minio://{bucket_name}/{object_name}"
            if session_id:
                chunk.metadata["session_id"] = session_id

        # Re-uploading the same file should REPLACE its old vectors, not add
        # a duplicate copy alongside them. auto_id=True means every insert
        # gets a fresh random PK, so Milvus has no idea two inserts are "the
        # same document" unless we tell it - delete any existing chunks for
        # this exact source (scoped to this session, so we never touch
        # another session's copy of a same-named file) before inserting.
        self.update_state(state="PROGRESS", meta={"msg": "Removing any previous version from VectorDB..."})
        source_id = f"minio://{bucket_name}/{object_name}"
        delete_expr = f"source == '{source_id}'"
        if session_id:
            delete_expr += f" && session_id == '{session_id}'"
        try:
            vector_store.delete(expr=delete_expr)
        except Exception as delete_err:
            # Don't fail ingestion just because there was nothing to delete
            # (e.g. first-ever upload of this file) or a transient issue -
            # log and continue; worst case this run behaves like before.
            print(f"[tasks] Warning: could not clear existing vectors for '{source_id}': {delete_err}")

        self.update_state(state="PROGRESS", meta={"msg": "Generating embeddings & storing in Milvus..."})
        vector_store.add_documents(documents=chunks)

        success_msg = f"Successfully loaded {object_name} from MinIO into VectorDB."
        _notify_ingestion_result(username, "Document ingestion complete", success_msg, "success",
                                 bucket_name, object_name)
        return {"status": "Success", "message": success_msg}
    except Exception as e:
        _notify_ingestion_result(username, "Document ingestion failed", str(e), "error",
                                 bucket_name, object_name)
        raise e


@celery_app.task
def expire_stale_sessions():
    """Celery Beat sweep: finds every AgentSessionModel that's been inactive
    longer than SESSION_TTL_DAYS and tears down ALL of its data -
    Postgres row (+ its chats, via ON DELETE CASCADE), and its Milvus
    document/memory vectors (services.purge_session_data). Redis short-term
    history isn't touched here - it already carries its own TTL
    (REDIS_HISTORY_TTL_SECONDS, set to match SESSION_TTL_DAYS in
    services.get_redis_history) and expires on its own.

    Must be run by a `celery beat` process on a schedule - see
    celery_app.conf.beat_schedule below and the `beat` service in
    docker-compose.yml. A plain `celery worker` will execute this task if
    asked to, but nothing will *ask* it to without beat running somewhere.
    """
    from datetime import datetime, timedelta, timezone
    from services import SessionLocal, purge_session_data, SESSION_TTL_DAYS
    from models import AgentSessionModel

    cutoff = datetime.now(timezone.utc) - timedelta(days=SESSION_TTL_DAYS)
    db = SessionLocal()
    expired_ids = []
    try:
        stale_sessions = db.query(AgentSessionModel).filter(AgentSessionModel.updated_at < cutoff).all()
        for session in stale_sessions:
            purge_session_data(session.id, username=session.username)
            db.delete(session)
            expired_ids.append(session.id)
        db.commit()
    finally:
        db.close()

    print(f"[tasks] expire_stale_sessions: purged {len(expired_ids)} session(s) older than "
         f"{SESSION_TTL_DAYS} day(s): {expired_ids}")
    return {"expired_count": len(expired_ids), "expired_session_ids": expired_ids}


# Runs expire_stale_sessions once an hour. Frequent enough that a 3-day
# local-testing TTL doesn't leave much stale data lying around, cheap enough
# (an indexed timestamp filter) not to matter at any real scale.
celery_app.conf.beat_schedule = {
    "expire-stale-sessions-hourly": {
        "task": "tasks.expire_stale_sessions",
        "schedule": 3600.0,
    },
}
celery_app.conf.timezone = "UTC"