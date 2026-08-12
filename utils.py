import os
from celery.result import AsyncResult

def check_task_status(task_id: str) -> str:
    from tasks import celery_app
    res = AsyncResult(task_id, app=celery_app)

    if res.state == "PENDING":
        return "Task is waiting in queue for a worker."
    elif res.state == "PROGRESS":
        msg = res.info.get("msg") if isinstance(res.info, dict) else str(res.info)
        return f"Task is currently processing. Progress metadata: {msg}"
    elif res.state == "SUCCESS":
        msg = res.result.get("message") if isinstance(res.result, dict) else str(res.result)
        return f"Task completed successfully! Output: {msg}"
    elif res.state == "FAILURE":
        return f"Task failed. Traceback/Error: {str(res.info)}"
    return f"Current status: {res.state}"

def query_rag_document(question: str) -> str:
    from services import retrieve_relevant_docs
    try:
        return retrieve_relevant_docs(question)
    except Exception as e:
        return f"Error executing RAG search: {str(e)}"