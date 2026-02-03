from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.evolution_engine import EvolutionEngine, get_evolution_session

router = APIRouter()

class ReapRequest(BaseModel):
    target: str
    scope: Optional[List[str]] = None

class ReapResponse(BaseModel):
    status: str
    target: str
    affected_chains: List[str]

@router.post("/reap", response_model=ReapResponse)
async def reap(
    req: ReapRequest,
    evolution: EvolutionEngine = Depends(get_evolution_session),
):
    result = await evolution.reap_chains(target=req.target, scope=req.scope or [])
    return ReapResponse(**result)