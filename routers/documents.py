import os
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from minio.error import S3Error
from sqlalchemy.orm import Session

from dependencies import get_db, get_current_user
from models import AgentSessionModel
from services import minio_client, trigger_ingestion_task

router = APIRouter(prefix="/documents", tags=["documents"])

DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "agent-documents")


def _ensure_bucket(bucket_name: str):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...),
                          bucket_name: str = Form(DEFAULT_BUCKET),
                          current_user: str = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    # Every document must be scoped to an owned, existing chat session -
    # this is what keeps Milvus's session_id partition (and therefore RAG
    # retrieval) private per user/session instead of one shared pool.
    # NOTE: this previously silently received `current_user` where
    # `session_id` belongs (trigger_ingestion_task's 3rd positional arg),
    # so uploads were never actually scoped to a session at all.
    session = db.query(AgentSessionModel).filter(AgentSessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Agent session not found.")
    if session.username != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to upload into this session.")

    safe_object_name = os.path.basename(file.filename or "")
    if not safe_object_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    storage_key = f"{session_id}/{safe_object_name}"

    try:
        _ensure_bucket(bucket_name)
        contents = await file.read()
        minio_client.put_object(
            bucket_name,
            storage_key,
            data=BytesIO(contents),
            length=len(contents),
            content_type=file.content_type or "application/octet-stream",
        )
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to store file in object storage: {e}")

    ingestion_result = trigger_ingestion_task(bucket_name, safe_object_name, session_id, current_user)
    return {
        "status": "uploaded",
        "bucket": bucket_name,
        "object_name": safe_object_name,
        "storage_key": ingestion_result["storage_key"],
        "session_id": session_id,
        "detail": ingestion_result["detail"],
    }