from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.evolution_engine import EvolutionEngine, get_evolution_session

router = APIRouter()

class EvolutionRequest(BaseModel):
    mode: str = "plan_only"  # plan_only | execute
    scope: List[str] = ["sdk", "backend", "db"]
    dry_run: bool = True

class EvolutionResponse(BaseModel):
    status: str
    actions: List[str]
    notes: Optional[str] = None

@router.post("/hyperspeed-evolution", response_model=EvolutionResponse)
async def hyperspeed_evolution(
    req: EvolutionRequest,
    evolution: EvolutionEngine = Depends(get_evolution_session),
):
    result = await evolution.run(
        mode=req.mode,
        scope=req.scope,
        dry_run=req.dry_run,
    )
    return EvolutionResponse(**result)