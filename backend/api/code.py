from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.grok_client import get_grok_client

router = APIRouter()

class CodeRequest(BaseModel):
    prompt: str
    language: Optional[str] = "python"
    constraints: Optional[Dict[str, Any]] = None

class CodeResponse(BaseModel):
    code: str
    explanation: Optional[str] = None
    reasoning_trace_id: Optional[str] = None

@router.post("/code", response_model=CodeResponse)
async def code(
    req: CodeRequest,
    grok = Depends(get_grok_client),
):
    result = await grok.generate_code(
        prompt=req.prompt,
        language=req.language,
        constraints=req.constraints or {},
    )
    return CodeResponse(**result)