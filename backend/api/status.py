from fastapi import APIRouter, Depends
from typing import Dict, Any

from services.quantum_engine import QuantumEngine, get_quantum_session
from services.evolution_engine import EvolutionEngine, get_evolution_session

router = APIRouter()

@router.get("/status")
async def status(
    quantum: QuantumEngine = Depends(get_quantum_session),
    evolution: EvolutionEngine = Depends(get_evolution_session),
):
    return {
        "service": "roboto-sai-backend",
        "version": "0.1.0",
        "quantum": quantum.get_status(),
        "evolution": evolution.status(),
    }