import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dependencies import get_db, get_current_user
from models import MCPServerModel
from schemas import MCPServerSchema

router = APIRouter(prefix="/mcp", tags=["mcp-servers"])


@router.get("", response_model=List[dict])
def list_mcp_servers(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    servers = db.query(MCPServerModel).all()
    return [{"server_id": s.server_id, "name": s.name, "command": s.command, "args": s.args, "env": s.env} for s in
            servers]


@router.post("", status_code=status.HTTP_201_CREATED)
def register_mcp_server(payload: MCPServerSchema, db: Session = Depends(get_db),
                        current_user: str = Depends(get_current_user)):
    server_id = payload.name.lower().replace(" ", "_")
    db_server = db.query(MCPServerModel).filter(MCPServerModel.server_id == server_id).first()

    if db_server:
        db_server.command = payload.command
        db_server.args = payload.args
        db_server.env = payload.env
    else:
        db_server = MCPServerModel(server_id=server_id, name=payload.name, command=payload.command, args=payload.args,
                                   env=payload.env)
        db.add(db_server)

    db.commit()
    return {"status": "success", "server_id": server_id}


@router.put("/{server_id}")
def edit_mcp_server(server_id: str, payload: MCPServerSchema, db: Session = Depends(get_db),
                    current_user: str = Depends(get_current_user)):
    db_server = db.query(MCPServerModel).filter(MCPServerModel.server_id == server_id).first()
    if not db_server:
        raise HTTPException(status_code=404, detail="MCP Server matrix entry missing.")

    db_server.name = payload.name
    db_server.command = payload.command
    db_server.args = payload.args
    db_server.env = payload.env
    db.commit()
    return {"status": "updated", "server_id": server_id}


@router.delete("/{server_id}")
def delete_mcp_server(server_id: str, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    db_server = db.query(MCPServerModel).filter(MCPServerModel.server_id == server_id).first()
    if not db_server:
        raise HTTPException(status_code=404, detail="MCP Server matrix entry missing.")

    db.delete(db_server)
    db.commit()
    return {"status": "deleted", "server_id": server_id}


@router.get("/{server_id}/tools")
async def list_tools_per_server(server_id: str, db: Session = Depends(get_db),
                                current_user: str = Depends(get_current_user)):
    srv = db.query(MCPServerModel).filter(MCPServerModel.server_id == server_id).first()
    if not srv:
        raise HTTPException(status_code=404, detail="Server matrix entry missing.")

    # StdioServerParameters.env REPLACES the subprocess's environment rather
    # than extending it - passing srv.env alone means the spawned server only
    # sees whatever was explicitly stored on the MCPServerModel row, not the
    # .env-derived vars (DATABASE_URL, AZURE_OPENAI_*, etc.) this process
    # already has. Merge our own environment in first so the subprocess
    # inherits everything by default, then let the row's env act as
    # per-server overrides/additions on top of that.
    merged_env = {**os.environ, **(srv.env or {})}
    server_params = StdioServerParameters(command=srv.command, args=srv.args, env=merged_env)
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                return {"server_id": server_id, "tools": tools_response.tools}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize or retrieve tools from MCP server '{server_id}': {str(e)}"
        )