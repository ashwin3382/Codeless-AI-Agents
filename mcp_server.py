import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["PYTHONWARNINGS"] = "ignore"

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from utils import check_task_status, query_rag_document
from fastmcp.server.dependencies import get_http_request
from services import trigger_ingestion_task


DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "agent-documents")

mcp = FastMCP("RAG Background Manager")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Plain, non-streaming endpoint for Docker/orchestrator healthchecks.
    """
    return PlainTextResponse("OK")


def _get_session_context():
    """session_id / username come from the SSE request's query string
    (set server-side by agent_engine.py from the real, verified session) -
    never from a tool argument, so an LLM (or a prompt-injected document)
    can't pick or override where its files land."""
    request = get_http_request()
    if not request:
        return None, None
    return request.query_params.get("session_id"), request.query_params.get("username")



@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """Queries the Milvus Vector DB using the strict document-bound LangChain RAG pipeline."""
    session_id, _ = _get_session_context()
    return query_rag_document(question, session_id)


@mcp.tool()
def ingest_minio_document(object_name: str) -> str:

    session_id, username = _get_session_context()
    if not session_id:
        return "Error: no active session context - ingestion must run inside an agent session."
    safe_object_name = os.path.basename(object_name or "")
    if not safe_object_name:
        return "Error: object_name is required and must not be empty."

    result = trigger_ingestion_task(DEFAULT_BUCKET, safe_object_name, session_id, username)
    return f"{result['detail']} Stored as: {result['storage_key']}"
@mcp.tool()
def check_ingestion_status(task_id: str) -> str:
    """Checks the live execution state of an ongoing MinIO document ingestion task."""
    return check_task_status(task_id)

if __name__ == "__main__":

    mcp.run(transport="sse", host="0.0.0.0", port=8000)