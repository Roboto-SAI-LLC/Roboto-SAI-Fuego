from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.grok_client import get_grok_client

router = APIRouter()

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
    return ChatResponse(
        reply=reply,
        reasoning_trace_id=meta.get("trace_id"),
        tokens_used=meta.get("tokens_used"),
        mode=meta.get("mode", "entangled"),
    )