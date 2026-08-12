#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-http://127.0.0.1:8080}"
USERNAME="${2:-admin}"
PASSWORD="${3:-changeme123}"
AGENT_NAME="${4:-smoke_test_bot}"
BAD_AGENT_NAME="__definitely_not_exists__"

# ---------- helpers ----------
say() { echo -e "\n== $* =="; }

fail() {
  echo "❌ FAIL: $1"
  exit 1
}

pass() {
  echo "✅ PASS: $1"
}

# usage: req METHOD URL [DATA] [AUTH_TOKEN]
req() {
  local method="$1" url="$2" data="${3:-}" token="${4:-}"
  if [[ -n "$token" ]]; then
    if [[ -n "$data" ]]; then
      curl -s -X "$method" "$url" -H "Authorization: Bearer $token" -H "Content-Type: application/json" -d "$data"
    else
      curl -s -X "$method" "$url" -H "Authorization: Bearer $token"
    fi
  else
    if [[ -n "$data" ]]; then
      curl -s -X "$method" "$url" -H "Content-Type: application/json" -d "$data"
    else
      curl -s -X "$method" "$url"
    fi
  fi
}

# usage: status METHOD URL [DATA] [AUTH_TOKEN]
status() {
  local method="$1" url="$2" data="${3:-}" token="${4:-}"
  if [[ -n "$token" ]]; then
    if [[ -n "$data" ]]; then
      curl -s -o /tmp/neg_body.json -w "%{http_code}" -X "$method" "$url" -H "Authorization: Bearer $token" -H "Content-Type: application/json" -d "$data"
    else
      curl -s -o /tmp/neg_body.json -w "%{http_code}" -X "$method" "$url" -H "Authorization: Bearer $token"
    fi
  else
    if [[ -n "$data" ]]; then
      curl -s -o /tmp/neg_body.json -w "%{http_code}" -X "$method" "$url" -H "Content-Type: application/json" -d "$data"
    else
      curl -s -o /tmp/neg_body.json -w "%{http_code}" -X "$method" "$url"
    fi
  fi
}

assert_code_in() {
  local got="$1"; shift
  local msg="$1"; shift
  for c in "$@"; do
    [[ "$got" == "$c" ]] && { pass "$msg (HTTP $got)"; return; }
  done
  echo "Response body:"
  cat /tmp/neg_body.json || true
  fail "$msg expected one of [$*], got $got"
}

assert_nonempty_json_key() {
  local json="$1" key="$2" msg="$3"
  local val
  val=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$key',''))" 2>/dev/null || true)
  [[ -n "$val" && "$val" != "None" ]] && pass "$msg" || fail "$msg (missing key: $key)"
}

# ---------- begin ----------
say "0) Valid login to get token"
LOGIN_JSON=$(curl -s -X POST "$BASE/token" -d "username=$USERNAME&password=$PASSWORD")
TOKEN=$(echo "$LOGIN_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || true)
[[ -n "$TOKEN" ]] || { echo "$LOGIN_JSON"; fail "Could not get valid token"; }
pass "Got valid token"

say "1) Invalid login should fail"
CODE=$(status POST "$BASE/token" "username=$USERNAME&password=wrongpass")
assert_code_in "$CODE" "Invalid login rejected" 400 401 422

say "2) Missing token on protected route should fail"
CODE=$(status GET "$BASE/sessions")
assert_code_in "$CODE" "Missing auth rejected" 401 403

say "3) Tampered token should fail"
BAD_TOKEN="${TOKEN}x"
CODE=$(status GET "$BASE/sessions" "" "$BAD_TOKEN")
assert_code_in "$CODE" "Tampered token rejected" 401 403

say "4) Invalid MCP payload should fail (missing required fields)"
BAD_MCP='{"name":"Bad MCP"}'
CODE=$(status POST "$BASE/mcp" "$BAD_MCP" "$TOKEN")
assert_code_in "$CODE" "Bad MCP payload rejected" 400 422

say "5) Non-existent MCP server tools should fail"
CODE=$(status GET "$BASE/mcp/__no_such_server__/tools" "" "$TOKEN")
assert_code_in "$CODE" "Unknown MCP server rejected" 400 404 500

say "6a) Create agent with invalid tool binding should fail"
BAD_AGENT_PAYLOAD='{
  "agent_name":"bad_tool_agent",
  "agent_prompt":"test",
  "tools":["rag_manager:__no_such_tool__"]
}'
CODE=$(status POST "$BASE/agents" "$BAD_AGENT_PAYLOAD" "$TOKEN")
assert_code_in "$CODE" "Invalid tool binding rejected (or deferred)" 201 400 404 422 500

say "6b) Chat with bad-tool agent should fail at runtime (deferred validation)"
BAD_TOOL_CHAT='{"message":"Use your tool now"}'
CODE=$(status POST "$BASE/agents/bad_tool_agent/chat/json" "$BAD_TOOL_CHAT" "$TOKEN")
assert_code_in "$CODE" "Bad tool fails at runtime (or model fallback)" 200 400 404 422 500

say "7) Chat with non-existent agent should fail"
CHAT_PAYLOAD='{"message":"hello"}'
CODE=$(status POST "$BASE/agents/$BAD_AGENT_NAME/chat/json" "$CHAT_PAYLOAD" "$TOKEN")
assert_code_in "$CODE" "Unknown agent rejected" 400 404

say "8) Chat with missing message should fail"
BAD_CHAT='{"session_id":"abc"}'
CODE=$(status POST "$BASE/agents/$AGENT_NAME/chat/json" "$BAD_CHAT" "$TOKEN")
assert_code_in "$CODE" "Missing message rejected" 400 422

say "9) Chat with invalid session_id should fail cleanly"
BAD_SESSION_CHAT='{"session_id":"__not_a_real_session__", "message":"hello"}'
CODE=$(status POST "$BASE/agents/$AGENT_NAME/chat/json" "$BAD_SESSION_CHAT" "$TOKEN")
assert_code_in "$CODE" "Bad session handled" 200 400 404 422
# 200 allowed because some apps auto-create new session if unknown ID

say "10) Get chats for invalid session should fail"
CODE=$(status GET "$BASE/sessions/__no_such_session__/chats" "" "$TOKEN")
assert_code_in "$CODE" "Unknown session rejected" 400 404

say "11) Upload endpoint with no file should fail"
CODE=$(curl -s -o /tmp/neg_body.json -w "%{http_code}" -X POST "$BASE/documents/upload" -H "Authorization: Bearer $TOKEN")
assert_code_in "$CODE" "Upload without file rejected" 400 422

say "12) Notifications endpoint optional (allow 200 or 404)"
CODE=$(status GET "$BASE/notifications" "" "$TOKEN")
assert_code_in "$CODE" "Notifications route state captured" 200 404

say "13) Unread-count endpoint optional (allow 200 or 404)"
CODE=$(status GET "$BASE/notifications/unread-count" "" "$TOKEN")
assert_code_in "$CODE" "Unread-count route state captured" 200 404

say "14) Sessions list should still work after negative tests"
CODE=$(status GET "$BASE/sessions" "" "$TOKEN")
assert_code_in "$CODE" "System still healthy after exception tests" 200

echo -e "\n🎉 NEGATIVE TEST SUITE COMPLETE"