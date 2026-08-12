import json
import uuid
from datetime import datetime, timezone
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from services import llm, get_redis_history, add_to_memory, retrieve_from_memory
from models import MCPServerModel, AgentSessionModel, AgentChatModel
from schemas import TelemetryReport, ToolExecutionTrace

PRICE_INPUT_PER_TOKEN = 2.50 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 10.00 / 1_000_000

MAX_AGENT_ITERATIONS = 8


async def run_agent_workflow(agent_cfg, payload, db, current_user, redis_client=None, pubsub_channel=None):
    if not payload.session_id:
        payload.session_id = uuid.uuid4().hex

    session_row = db.query(AgentSessionModel).filter(AgentSessionModel.id == payload.session_id).first()
    if session_row is None:
        session_row = AgentSessionModel(
            id=payload.session_id,
            agent_name=agent_cfg.agent_name,
            username=current_user,
            title=payload.message[:60],
        )
        db.add(session_row)
        db.commit()

    system_instruction = f"System Prompt:\n{agent_cfg.agent_prompt}\n\nGuardrails:\n{agent_cfg.guardrails or ''}"
    history_manager = get_redis_history(payload.session_id)

    base_messages = [SystemMessage(content=system_instruction)]
    short_term_messages = list(history_manager.messages)

    if not short_term_messages:
        long_term_messages = retrieve_from_memory(payload.session_id, query=payload.message)
        if long_term_messages:
            base_messages.append(SystemMessage(
                content="The following are relevant earlier exchanges with this user, "
                        "recalled from long-term memory (the live session had expired):"
            ))
            base_messages.extend(long_term_messages)

    for msg in short_term_messages:
        if msg.type == "human":
            base_messages.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            base_messages.append(AIMessage(content=msg.content))
    base_messages.append(HumanMessage(content=payload.message))

    assigned_tools_raw = agent_cfg.tools or []
    needed_servers = set(t.split(":")[0] for t in assigned_tools_raw if ":" in t)
    server_configs = db.query(MCPServerModel).filter(MCPServerModel.server_id.in_(needed_servers)).all()
    server_map = {srv.server_id: srv for srv in server_configs}

    telemetry = TelemetryReport()
    ai_response_content = ""

    async with AsyncExitStack() as stack:
        mcp_sessions = {}

        for srv_id in needed_servers:
            if srv_id in server_map:
                cfg = server_map[srv_id]
                params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env)
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                mcp_sessions[srv_id] = session

        openai_tools = []
        tools_cache = {}
        for tool_str in assigned_tools_raw:
            if ":" in tool_str:
                srv_id, tool_name = tool_str.split(":", 1)
                if srv_id in mcp_sessions:
                    if srv_id not in tools_cache:
                        tools_cache[srv_id] = (await mcp_sessions[srv_id].list_tools()).tools
                    for t in tools_cache[srv_id]:
                        if t.name == tool_name:
                            openai_tools.append({
                                "type": "function",
                                "function": {"name": f"{srv_id}__{t.name}", "description": t.description,
                                             "parameters": t.inputSchema}
                            })

        runnable_llm = llm.bind_tools(openai_tools) if openai_tools else llm
        current_messages = list(base_messages)

        for _ in range(MAX_AGENT_ITERATIONS):
            tool_calls = None
            iteration_text = ""

            if redis_client and pubsub_channel:
                accumulated_chunk = None
                async for chunk in runnable_llm.astream(current_messages):
                    if chunk.content:
                        iteration_text += chunk.content
                        await redis_client.publish(pubsub_channel,
                                                   json.dumps({'event': 'token', 'text': chunk.content}))
                    accumulated_chunk = chunk if accumulated_chunk is None else accumulated_chunk + chunk

                if accumulated_chunk:
                    tool_calls = getattr(accumulated_chunk, "tool_calls", None)
                    usage = getattr(accumulated_chunk, "usage_metadata", None) or {}
                    telemetry.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    telemetry.total_completion_tokens += usage.get("completion_tokens", 0)
            else:
                chunk = await runnable_llm.ainvoke(current_messages)
                usage = getattr(chunk, "usage_metadata", None) or getattr(chunk, "response_metadata", {}).get(
                    "token_usage", {})
                telemetry.total_prompt_tokens += usage.get("prompt_tokens", 0)
                telemetry.total_completion_tokens += usage.get("completion_tokens", 0)
                if chunk.content:
                    iteration_text += chunk.content
                tool_calls = getattr(chunk, "tool_calls", None)

            ai_response_content += iteration_text

            if tool_calls:
                current_messages.append(AIMessage(content=iteration_text, tool_calls=tool_calls))
                for call in tool_calls:
                    if "__" in call["name"]:
                        srv_id, real_tool_name = call["name"].split("__", 1)
                        if srv_id in mcp_sessions:
                            trace = ToolExecutionTrace(tool_name=real_tool_name, input_arguments=call["args"],
                                                       output_data="")
                            if redis_client and pubsub_channel:
                                await redis_client.publish(pubsub_channel,
                                                           json.dumps({'event': 'tool_start', 'tool': real_tool_name}))

                            mcp_result = await mcp_sessions[srv_id].call_tool(real_tool_name, call["args"])
                            tool_output = "".join([c.text for c in mcp_result.content if hasattr(c, 'text')])

                            trace.output_data = tool_output
                            telemetry.tool_calls_executed.append(trace)
                            current_messages.append(ToolMessage(content=tool_output, tool_call_id=call["id"]))
            else:
                break
        else:
            if not ai_response_content:
                ai_response_content = ("I wasn't able to finish this within the allowed number of tool "
                                       "round-trips. Please try rephrasing your request.")

    telemetry.total_token_usage = telemetry.total_prompt_tokens + telemetry.total_completion_tokens
    telemetry.total_estimated_cost_usd = (telemetry.total_prompt_tokens * PRICE_INPUT_PER_TOKEN) + (
                telemetry.total_completion_tokens * PRICE_OUTPUT_PER_TOKEN)

    history_manager.add_user_message(payload.message)
    history_manager.add_ai_message(ai_response_content)

    add_to_memory(payload.session_id, payload.message, ai_response_content)

    db.add(AgentChatModel(session_id=payload.session_id, role="human", content=payload.message))
    db.add(AgentChatModel(
        session_id=payload.session_id,
        role="ai",
        content=ai_response_content,
        prompt_tokens=telemetry.total_prompt_tokens,
        completion_tokens=telemetry.total_completion_tokens,
        estimated_cost_usd=telemetry.total_estimated_cost_usd,
    ))
    session_row.updated_at = datetime.now(timezone.utc)
    db.commit()

    return ai_response_content, telemetry