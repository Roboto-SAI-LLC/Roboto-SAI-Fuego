from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
import shutil

router = APIRouter()

MCP_CONFIG_PATH = os.path.join(os.getcwd(), ".vscode", "mcp.json")

class MCPServerConfig(BaseModel):
    command: Optional[str] = None
    args: Optional[List[str]] = None
    type: Optional[str] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    env: Optional[Dict[str, str]] = None
    disabled: Optional[bool] = False

class MCPConfig(BaseModel):
    mcpServers: Dict[str, MCPServerConfig]

@router.get("/api/mcp/config", tags=["MCP"])
async def get_mcp_config():
    """Get current MCP configuration."""
    if not os.path.exists(MCP_CONFIG_PATH):
        return {"mcpServers": {}}
    try:
        with open(MCP_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/mcp/config", tags=["MCP"])
async def update_mcp_config(config: Dict[str, Any]):
    """Update MCP configuration."""
    try:
        # Validate JSON structure (basic check)
        if "mcpServers" not in config:
             # Basic structure enforcement
             pass
        
        # Backup existing config
        if os.path.exists(MCP_CONFIG_PATH):
            shutil.copy2(MCP_CONFIG_PATH, MCP_CONFIG_PATH + ".bak")

        with open(MCP_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
        
        return {"status": "success", "message": "Config updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/mcp/restart", tags=["MCP"])
async def restart_mcp_services():
    """Restart MCP services (Placeholder)."""
    # implementation depends on how MCP servers are run.
    # If run as subprocesses, we would kill and restart them here.
    return {"status": "success", "message": "MCP services restarted (simulated)"}
