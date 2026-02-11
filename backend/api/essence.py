from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.essence_store import EssenceStore, get_essence_session

router = APIRouter()

class EssenceStoreRequest(BaseModel):
    essence: Dict[str, Any]
    category: str
    tags: Optional[List[str]] = None
    workspace_id: Optional[str] = None

class EssenceStoreResponse(BaseModel):
    id: str
    category: str
    stored_at: str
    embedding_id: Optional[str] = None

class EssenceRetrieveResponse(BaseModel):
    items: List[Dict[str, Any]]

@router.post("/essence/store", response_model=EssenceStoreResponse)
async def essence_store(
    req: EssenceStoreRequest,
    store: EssenceStore = Depends(get_essence_session),
):
    if store is None:
        raise HTTPException(status_code=503, detail="Essence store not initialized")
    import json
    record = store.store_essence(
        data=json.dumps(req.essence),
        category=req.category,
        tags=req.tags,
    )
    if not record.get("success"):
        raise HTTPException(status_code=500, detail=record.get("error", "Store failed"))
    return EssenceStoreResponse(
        id=record.get("essence_id", "unknown"),
        category=record.get("category", req.category),
        stored_at=str(record.get("timestamp", "")),
        embedding_id=None,
    )

@router.get("/essence/retrieve", response_model=EssenceRetrieveResponse)
async def essence_retrieve(
    category: Optional[str] = None,
    workspace_id: Optional[str] = None,
    limit: int = 20,
    store: EssenceStore = Depends(get_essence_session),
):
    if store is None:
        raise HTTPException(status_code=503, detail="Essence store not initialized")
    items = store.retrieve_essence(
        category=category or "general",
        tags=None,
        limit=limit,
    )
    return EssenceRetrieveResponse(items=items)