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
def process_minio_document_bg(self, bucket_name: str, object_name: str, username: str = None):
    """Downloads files from MinIO storage and handles chunk parsing in a background worker."""
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
            # the source, don't wipe everything else with .clear().
            chunk.metadata["source"] = f"minio://{bucket_name}/{object_name}"

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
