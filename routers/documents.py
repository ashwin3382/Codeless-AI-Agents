import os
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from minio.error import S3Error

from dependencies import get_current_user
from services import minio_client, trigger_ingestion_task

router = APIRouter(prefix="/documents", tags=["documents"])

DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "agent-documents")


def _ensure_bucket(bucket_name: str):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), bucket_name: str = Form(DEFAULT_BUCKET),
                          current_user: str = Depends(get_current_user)):
    safe_object_name = os.path.basename(file.filename or "")
    if not safe_object_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    try:
        _ensure_bucket(bucket_name)
        contents = await file.read()
        minio_client.put_object(
            bucket_name,
            safe_object_name,
            data=BytesIO(contents),
            length=len(contents),
            content_type=file.content_type or "application/octet-stream",
        )
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to store file in object storage: {e}")

    ingestion_message = trigger_ingestion_task(bucket_name, safe_object_name, current_user)
    return {"status": "uploaded", "bucket": bucket_name, "object_name": safe_object_name, "detail": ingestion_message}