from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
import json
import logging
import re
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from supabase._async.client import AsyncClient

from services.grok_client import get_grok_client
from utils.supabase_client import get_async_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Supabase dependency ──────────────────────────────────────────────

async def _get_supabase() -> AsyncClient:
    client = await get_async_supabase_client()
    if not client:
        raise HTTPException(status_code=500, detail="Supabase client not available")
    return client


async def _get_current_user(request: Request, supabase: AsyncClient = Depends(_get_supabase)) -> dict:
    """Extract and verify the authenticated user from access_token cookie or Authorization header."""
    auth_header = request.headers.get("authorization")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    else:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_response = await supabase.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"id": user_response.user.id, "email": user_response.user.email}
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Auth failed: {exc}")


class ToolRequest(BaseModel):
    serverId: Optional[str] = Field(
        default=None,
        description="Optional MCP server identifier; defaults to the filesystem server."
    )
    toolName: str
    description: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class ChatEvent(BaseModel):
    id: str
    timestamp: int
    type: str
    data: Dict[str, Any]


class ChatRequest(BaseModel):
    message: str
    reasoning_effort: Optional[str] = Field(default="medium")
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None  # For memory integration
    session_id: Optional[str] = None  # For conversation continuity
    tool_request: Optional[ToolRequest] = None


class ChatResponse(BaseModel):
    reply: str
    reasoning_trace_id: Optional[str] = None
    tokens_used: Optional[int] = None
    mode: Optional[str] = "entangled"
    events: List[ChatEvent] = Field(default_factory=list)


def _current_millis() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _create_event(event_type: str, data: Dict[str, Any]) -> ChatEvent:
    return ChatEvent(
        id=str(uuid4()),
        timestamp=_current_millis(),
        type=event_type,
        data=data,
    )


def _extract_path(message: str) -> Optional[str]:
    match = re.search(r"[A-Za-z]:\\[^,;\n]+", message)
    if match:
        return match.group(0)
    return None


def _parse_inline_tool_request(message: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"```tool_request\s*\n([\s\S]+?)\n```", message, flags=re.IGNORECASE)
    if not match:
        return None
    payload = match.group(1).strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _context_tool_request(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not context:
        return None
    for key in ("tool_request", "toolCall", "mcp_tool", "mcp_tool_call"):
        candidate = context.get(key)
        if isinstance(candidate, dict) and candidate.get("toolName"):
            return candidate
    return None


def _match_heuristic(message: str) -> Optional[ToolRequest]:
    normalized = message.lower()
    path = _extract_path(message)
    if not path:
        path = "D:\\temp.txt"

    if any(phrase in normalized for phrase in ["write file", "save file", "create file"]):
        return ToolRequest(
            serverId="filesystem",
            toolName="fs_writeFile",
            description="Write a file using MCP",
            args={
                "path": path,
                "content": message,
                "createDirs": True,
            },
        )

    if any(phrase in normalized for phrase in ["read file", "open file", "show file"]):
        return ToolRequest(
            serverId="filesystem",
            toolName="fs_readFile",
            description="Read a file via MCP",
            args={
                "path": path,
                "encoding": "utf-8",
            },
        )

    if any(phrase in normalized for phrase in ["list directory", "list files", "show folder"]):
        directory = path if path and path.endswith("\\") else (path + "\\" if path else "D:\\")
        return ToolRequest(
            serverId="filesystem",
            toolName="fs_listDir",
            description="List directory contents",
            args={
                "path": directory,
            },
        )

    if "tweet" in normalized and any(phrase in normalized for phrase in ["post", "publish"]):
        return ToolRequest(
            serverId="twitter",
            toolName="twitter_postTweet",
            description="Post to Twitter via MCP",
            args={
                "content": message,
            },
        )

    if "email" in normalized and "send" in normalized:
        return ToolRequest(
            serverId="email",
            toolName="email_sendEmail",
            description="Send email via MCP",
            args={
                "to": "demo@example.com",
                "subject": "Requested via Roboto SAI",
                "body": message,
            },
        )

    return None


def _build_tool_request(req: ChatRequest) -> Optional[ToolRequest]:
    explicit = req.tool_request
    if explicit:
        return explicit

    inline = _parse_inline_tool_request(req.message)
    if inline:
        try:
            return ToolRequest(**inline)
        except Exception:  # pragma: no cover - best effort
            pass

    context_request = _context_tool_request(req.context)
    if context_request:
        try:
            return ToolRequest(**context_request)
        except Exception:
            pass

    return _match_heuristic(req.message)


def _build_tool_event(tool_request: ToolRequest) -> ChatEvent:
    data = {
        "id": str(uuid4()),
        "source": "backend",
        "serverId": tool_request.serverId or "filesystem",
        "toolName": tool_request.toolName,
        "description": tool_request.description or "Requires MCP execution",
        "args": tool_request.args,
    }
    return _create_event("tool_call", data)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    grok = Depends(get_grok_client),
):
    reply, meta = await grok.chat(
        message=req.message,
        reasoning_effort=req.reasoning_effort,
        context=req.context or {},
        user_id=req.user_id,
        session_id=req.session_id,
    )

    assistant_event = _create_event("assistant_message", {
        "content": reply,
        "metadata": {
            "reasoning_trace_id": meta.get("trace_id"),
            "mode": meta.get("mode"),
            "elapsed": meta.get("elapsed"),
        },
    })

    events = [assistant_event]
    tool_request = _build_tool_request(req)
    if tool_request:
        events.append(_build_tool_event(tool_request))

    return ChatResponse(
        reply=reply,
        reasoning_trace_id=meta.get("trace_id"),
        tokens_used=meta.get("tokens_used"),
        mode=meta.get("mode", "entangled"),
        events=events,
    )


# ── Chat history ─────────────────────────────────────────────────────

@router.get("/chat/history")
async def get_chat_history(
    request: Request,
    session_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    user: dict = Depends(_get_current_user),
    supabase: AsyncClient = Depends(_get_supabase),
):
    """Retrieve recent chat history for the authenticated user."""
    try:
        query = (
            supabase.table("messages")
            .select("*")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .limit(limit)
        )
        if session_id:
            query = query.eq("session_id", session_id)

        result = await query.execute()
        messages = result.data or []

        return {
            "success": True,
            "count": len(messages),
            "messages": [
                {
                    "id": msg.get("id"),
                    "user_id": msg.get("user_id"),
                    "session_id": msg.get("session_id"),
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "emotion": msg.get("emotion"),
                    "emotion_text": msg.get("emotion_text"),
                    "emotion_probabilities": (
                        json.loads(msg["emotion_probabilities"])
                        if isinstance(msg.get("emotion_probabilities"), str)
                        else msg.get("emotion_probabilities")
                    ),
                    "created_at": msg.get("created_at"),
                }
                for msg in messages
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.error(f"Chat history error: {exc}")
        # Return empty rather than crash — table may not exist yet
        return {
            "success": True,
            "count": 0,
            "messages": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Feedback ─────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Message feedback request"""
    message_id: str
    rating: int  # 1=thumbs up, -1=thumbs down


@router.post("/chat/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    user: dict = Depends(_get_current_user),
    supabase: AsyncClient = Depends(_get_supabase),
):
    """Submit thumbs up/down feedback for a message."""
    if feedback.rating not in [1, -1]:
        raise HTTPException(status_code=400, detail="Rating must be 1 (thumbs up) or -1 (thumbs down)")

    try:
        _uuid.UUID(feedback.message_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid message_id format")

    data = {
        "message_id": feedback.message_id,
        "user_id": user["id"],
        "rating": feedback.rating,
    }
    try:
        await supabase.table("message_feedback").insert(data).execute()
    except Exception as exc:
        logger.error(f"Feedback insert error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

    return {
        "success": True,
        "message": "Feedback recorded. The eternal flame adapts.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
