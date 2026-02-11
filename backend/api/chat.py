from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
import base64
import json
import logging
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


# TODO: Add authentication dependency (user: dict = Depends(_get_current_user))
# Currently unauthenticated for compatibility with unauthenticated frontends
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

    return ChatResponse(
        reply=reply,
        reasoning_trace_id=meta.get("trace_id"),
        tokens_used=meta.get("tokens_used"),
        mode=meta.get("mode", "entangled"),
        events=[assistant_event],
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


# ── Sessions ─────────────────────────────────────────────────────────

@router.get("/sessions")
async def get_sessions(
    limit: int = Query(20, ge=1, le=100),
    user: dict = Depends(_get_current_user),
    supabase: AsyncClient = Depends(_get_supabase),
):
    """List user's chat sessions with metadata."""
    try:
        # Get all messages for the user
        result = await (
            supabase.table("messages")
            .select("session_id, created_at, role, content")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .execute()
        )
        
        messages = result.data or []
        
        # Group by session_id
        sessions_map = {}
        for msg in messages:
            sid = msg.get("session_id")
            if not sid:
                continue
            
            if sid not in sessions_map:
                sessions_map[sid] = {
                    "session_id": sid,
                    "message_count": 0,
                    "first_message_time": msg.get("created_at"),
                    "last_message_time": msg.get("created_at"),
                    "preview": None,
                }
            
            sessions_map[sid]["message_count"] += 1
            
            # Update first/last message times
            current_first = sessions_map[sid]["first_message_time"]
            current_last = sessions_map[sid]["last_message_time"]
            msg_time = msg.get("created_at")
            
            if msg_time:
                if current_first is None or msg_time < current_first:
                    sessions_map[sid]["first_message_time"] = msg_time
                if current_last is None or msg_time > current_last:
                    sessions_map[sid]["last_message_time"] = msg_time
            
            # Set preview to first user message
            if msg.get("role") == "user" and not sessions_map[sid]["preview"]:
                content = msg.get("content", "")
                sessions_map[sid]["preview"] = content[:100] + ("..." if len(content) > 100 else "")
        
        # Convert to list and sort by last message time
        sessions_list = list(sessions_map.values())
        sessions_list.sort(key=lambda s: s["last_message_time"] or "", reverse=True)
        
        # Apply limit
        sessions_list = sessions_list[:limit]
        
        return {
            "success": True,
            "count": len(sessions_list),
            "sessions": sessions_list,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.error(f"Sessions list error: {exc}")
        return {
            "success": True,
            "count": 0,
            "sessions": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Conversation Rollup ──────────────────────────────────────────────

@router.get("/conversations/rollup")
async def get_conversation_rollup(
    session_id: str = Query(..., description="Session ID to get rollup for"),
    user: dict = Depends(_get_current_user),
    supabase: AsyncClient = Depends(_get_supabase),
):
    """Get summary rollup for a conversation session."""
    try:
        # Verify session belongs to user
        msg_check = await (
            supabase.table("messages")
            .select("id")
            .eq("session_id", session_id)
            .eq("user_id", user["id"])
            .limit(1)
            .execute()
        )
        
        if not msg_check.data:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
        
        # Get rollup from conversation_rollups table
        result = await (
            supabase.table("conversation_rollups")
            .select("*")
            .eq("session_id", session_id)
            .eq("user_id", user["id"])
            .maybe_single()
            .execute()
        )
        
        if not result.data:
            return {
                "success": True,
                "exists": False,
                "message": "No rollup available. Generate one using POST /conversations/summarize",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        rollup = result.data
        
        return {
            "success": True,
            "exists": True,
            "rollup": {
                "session_id": rollup.get("session_id"),
                "summary": rollup.get("summary"),
                "key_topics": (
                    json.loads(rollup["key_topics"])
                    if isinstance(rollup.get("key_topics"), str)
                    else rollup.get("key_topics")
                ),
                "sentiment": rollup.get("sentiment"),
                "message_count": rollup.get("message_count"),
                "created_at": rollup.get("created_at"),
                "updated_at": rollup.get("updated_at"),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Rollup fetch error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch rollup")


# ── Summarize Conversation ───────────────────────────────────────────

@router.post("/conversations/summarize")
async def summarize_conversation(
    session_id: str = Query(..., description="Session ID to summarize"),
    force: bool = Query(False, description="Force regenerate even if summary exists"),
    user: dict = Depends(_get_current_user),
    supabase: AsyncClient = Depends(_get_supabase),
    grok = Depends(get_grok_client),
):
    """Generate AI summary for a conversation session."""
    try:
        # Get all messages for the session
        result = await (
            supabase.table("messages")
            .select("*")
            .eq("session_id", session_id)
            .eq("user_id", user["id"])
            .order("created_at", desc=False)
            .execute()
        )
        
        messages = result.data or []
        
        if not messages:
            raise HTTPException(status_code=404, detail="Session not found or has no messages")
        
        # Check if rollup already exists
        if not force:
            existing = await (
                supabase.table("conversation_rollups")
                .select("id")
                .eq("session_id", session_id)
                .eq("user_id", user["id"])
                .maybe_single()
                .execute()
            )
            
            if existing.data:
                return {
                    "success": True,
                    "message": "Summary already exists. Use force=true to regenerate.",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        
        # Build conversation context for Grok
        conversation_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text.append(f"{role.upper()}: {content}")
        
        conversation_str = "\n".join(conversation_text)
        
        # Generate summary using Grok
        summary_prompt = f"""Analyze the following conversation and provide:
1. A concise summary (2-3 sentences)
2. Key topics discussed (as a JSON array of strings)
3. Overall sentiment (positive, neutral, or negative)

Conversation:
{conversation_str}

Respond in JSON format:
{{
  "summary": "...",
  "key_topics": ["topic1", "topic2", ...],
  "sentiment": "positive/neutral/negative"
}}"""
        
        reply, meta = await grok.chat(
            message=summary_prompt,
            reasoning_effort="low",
            context={},
            user_id=user["id"],
            session_id=None,  # Don't save this meta-conversation
        )
        
        # Parse Grok response
        try:
            # Extract JSON from response (handle code blocks)
            json_start = reply.find("{")
            json_end = reply.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = reply[json_start:json_end]
                summary_data = json.loads(json_str)
            else:
                # Fallback if JSON not found
                summary_data = {
                    "summary": reply[:500],
                    "key_topics": [],
                    "sentiment": "neutral"
                }
        except json.JSONDecodeError:
            summary_data = {
                "summary": reply[:500],
                "key_topics": [],
                "sentiment": "neutral"
            }
        
        # Upsert to conversation_rollups
        rollup_data = {
            "session_id": session_id,
            "user_id": user["id"],
            "summary": summary_data.get("summary", "")[:1000],  # Limit length
            "key_topics": json.dumps(summary_data.get("key_topics", [])),
            "sentiment": summary_data.get("sentiment", "neutral"),
            "message_count": len(messages),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Check if exists to decide insert vs update
        existing_check = await (
            supabase.table("conversation_rollups")
            .select("id")
            .eq("session_id", session_id)
            .eq("user_id", user["id"])
            .maybe_single()
            .execute()
        )
        
        if existing_check.data:
            # Update
            await (
                supabase.table("conversation_rollups")
                .update(rollup_data)
                .eq("session_id", session_id)
                .eq("user_id", user["id"])
                .execute()
            )
        else:
            # Insert
            rollup_data["created_at"] = datetime.now(timezone.utc).isoformat()
            await supabase.table("conversation_rollups").insert(rollup_data).execute()
        
        return {
            "success": True,
            "message": "Summary generated successfully",
            "session_id": session_id,
            "summary": summary_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Summarize error: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(exc)}")


# ── Enhanced Chat History with Cursor Pagination ─────────────────────

@router.get("/chat/history/paginated")
async def get_chat_history_paginated(
    request: Request,
    session_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    cursor: Optional[str] = Query(None, description="Pagination cursor from previous response"),
    user: dict = Depends(_get_current_user),
    supabase: AsyncClient = Depends(_get_supabase),
):
    """Retrieve chat history with cursor-based pagination."""
    try:
        # Decode cursor if provided
        after_timestamp = None
        if cursor:
            try:
                decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
                after_timestamp = decoded
            except Exception as e:
                logger.warning(f"Invalid cursor: {e}")
                raise HTTPException(status_code=400, detail="Invalid cursor")
        
        # Build query
        query = (
            supabase.table("messages")
            .select("*")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .limit(limit + 1)  # Fetch one extra to determine if there's more
        )
        
        if session_id:
            query = query.eq("session_id", session_id)
        
        if after_timestamp:
            query = query.lt("created_at", after_timestamp)
        
        result = await query.execute()
        messages = result.data or []
        
        # Determine if there are more results
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]
        
        # Generate next cursor
        next_cursor = None
        if has_more and messages:
            last_timestamp = messages[-1].get("created_at")
            if last_timestamp:
                next_cursor = base64.urlsafe_b64encode(last_timestamp.encode()).decode()
        
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
            "pagination": {
                "has_more": has_more,
                "next_cursor": next_cursor,
                "limit": limit,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Paginated history error: {exc}")
        return {
            "success": True,
            "count": 0,
            "messages": [],
            "pagination": {
                "has_more": False,
                "next_cursor": None,
                "limit": limit,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
