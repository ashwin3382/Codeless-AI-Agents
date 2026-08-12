import warnings

# Keep noisy deprecation/user warnings out of the process entirely - useful
# hygiene even on stdio transport, since anything unexpected on stdout could
# corrupt the MCP protocol stream.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["PYTHONWARNINGS"] = "ignore"

from fastmcp import FastMCP
from utils import check_task_status, query_rag_document
from tasks import process_minio_document_bg

mcp = FastMCP("RAG Background Manager")


@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """Queries the Milvus Vector DB using the strict document-bound LangChain RAG pipeline."""
    return query_rag_document(question)


@mcp.tool()
def ingest_minio_document(bucket_name: str, object_name: str) -> str:
    """Triggers a background Celery task to download, split, embed, and ingest an object from MinIO."""
    task = process_minio_document_bg.delay(bucket_name, object_name)
    return f"Ingestion task started successfully! Task ID: {task.id}."


@mcp.tool()
def check_ingestion_status(task_id: str) -> str:
    """Checks the live execution state of an ongoing MinIO document ingestion task."""
    return check_task_status(task_id)


if __name__ == "__main__":
    # Runs over stdio so it can be spawned as a subprocess via
    # StdioServerParameters (see agent_engine.py / main.py), matching how
    # MCPServerModel rows are configured (command="python", args=["mcp_server.py"]).
    # Previously this set sys.argv for a stdio run but then called
    # mcp.run(transport="http", ...) anyway, so the two never matched.
    mcp.run(transport="stdio")
