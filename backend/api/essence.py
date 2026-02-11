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
    record = await store.store_essence(
        essence=req.essence,
        category=req.category,
        tags=req.tags or [],
        workspace_id=req.workspace_id,
    )
    return EssenceStoreResponse(**record)

@router.get("/essence/retrieve", response_model=EssenceRetrieveResponse)
async def essence_retrieve(
    category: Optional[str] = None,
    workspace_id: Optional[str] = None,
    limit: int = 20,
    store: EssenceStore = Depends(get_essence_session),
):
    items = await store.retrieve_essence(
        category=category,
        workspace_id=workspace_id,
        limit=limit,
    )
    return EssenceRetrieveResponse(items=items)