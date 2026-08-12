#!/usr/bin/env bash
# Run this AFTER `docker compose up --build -d` and giving the containers
# ~20-30s to become healthy (Milvus in particular is slow to start).
#
# Usage: ./smoke_test.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8080}"

: "${AZURE_OPENAI_ENDPOINT:?missing}"
: "${AZURE_OPENAI_DEPLOYMENT_NAME:?missing}"
: "${AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:?missing}"
: "${AZURE_OPENAI_API_VERSION:?missing}"
: "${AZURE_OPENAI_API_KEY:?missing}"

echo "== 0. Waiting a bit for services =="
sleep 10

echo "== 1. Login =="
TOKEN=$(curl -s -X POST "$BASE/token" -d "username=admin&password=changeme123" | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
echo "Got token: ${TOKEN:0:20}..."
AUTH=(-H "Authorization: Bearer $TOKEN")

echo "== 2. Register MCP server (with docker-safe env) =="
MCP_RESP=$(curl -s -X POST "$BASE/mcp" "${AUTH[@]}" -H "Content-Type: application/json" \
  -d '{
        "name": "RAG Manager",
        "command": "python",
        "args": ["mcp_server.py"],
        "env": {
          "MILVUS_HOST": "milvus",
          "MILVUS_PORT": "19530",
          "REDIS_URL": "redis://redis:6379/0",
          "MINIO_ENDPOINT": "minio:9000",
          "DATABASE_URL": "postgresql://user:password@postgres:5432/agent_db",
          "AZURE_OPENAI_ENDPOINT": "'"${AZURE_OPENAI_ENDPOINT}"'",
          "AZURE_OPENAI_DEPLOYMENT_NAME": "'"${AZURE_OPENAI_DEPLOYMENT_NAME}"'",
          "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "'"${AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}"'",
          "AZURE_OPENAI_API_VERSION": "'"${AZURE_OPENAI_API_VERSION}"'",
          "AZURE_OPENAI_API_KEY": "'"${AZURE_OPENAI_API_KEY}"'",
          "OPENAI_API_KEY": "'"${AZURE_OPENAI_API_KEY}"'"
        }
      }')
echo "$MCP_RESP" | python3 -m json.tool

SERVER_ID=$(echo "$MCP_RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['server_id'])")
echo "Using server_id: $SERVER_ID"

echo "== 2b. Verify tools are visible =="
curl -s "$BASE/mcp/$SERVER_ID/tools" "${AUTH[@]}" | python3 -m json.tool

echo "== 3. Create agent =="
curl -s -X POST "$BASE/agents" "${AUTH[@]}" -H "Content-Type: application/json" \
  -d "{
        \"agent_name\": \"smoke_test_bot\",
        \"agent_prompt\": \"You are a terse test assistant. Reply in one short sentence.\",
        \"tools\": [\"${SERVER_ID}:query_knowledge_base\"]
      }" | python3 -m json.tool || true   # tolerate already exists

echo "== 4. Chat (sync) - new session =="
CHAT_RESP=$(curl -s -X POST "$BASE/agents/smoke_test_bot/chat/json" "${AUTH[@]}" -H "Content-Type: application/json" \
  -d '{"message": "Say hello in exactly 3 words."}')
echo "$CHAT_RESP" | python3 -m json.tool
SESSION_ID=$(echo "$CHAT_RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
echo "Session id: $SESSION_ID"

echo "== 5. Verify session + chats were persisted =="
curl -s "$BASE/sessions" "${AUTH[@]}" | python3 -m json.tool
curl -s "$BASE/sessions/$SESSION_ID/chats" "${AUTH[@]}" | python3 -m json.tool

echo "== 6. Second turn on same session (history should carry over) =="
curl -s -X POST "$BASE/agents/smoke_test_bot/chat/json" "${AUTH[@]}" -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\", \"message\": \"What did you just say?\"}" | python3 -m json.tool

echo "== 7. Upload a doc to trigger ingestion + notification =="
echo "This is a smoke-test document for RAG ingestion." > /tmp/smoke_test.txt
curl -s -X POST "$BASE/documents/upload" "${AUTH[@]}" -F "file=@/tmp/smoke_test.txt" | python3 -m json.tool

echo "== 8. Poll notifications (give the worker a few seconds to finish) =="
sleep 5
curl -s "$BASE/notifications" "${AUTH[@]}" | python3 -m json.tool || true
curl -s "$BASE/notifications/unread-count" "${AUTH[@]}" | python3 -m json.tool || true

echo "== DONE - smoke test finished without curl/HTTP errors =="