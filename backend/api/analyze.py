from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.grok_client import get_grok_client

router = APIRouter()

class AnalyzeRequest(BaseModel):
    text: str
    analysis_mode: str = "multi-layer"
    dimensions: Optional[List[str]] = None

class AnalyzeResponse(BaseModel):
    summary: str
    layers: List[Dict[str, Any]]
    confidence: float

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest,
    grok = Depends(get_grok_client),
):
    result = await grok.analyze(
        text=req.text,
        mode=req.analysis_mode,
        dimensions=req.dimensions or [],
    )
    return AnalyzeResponse(**result)