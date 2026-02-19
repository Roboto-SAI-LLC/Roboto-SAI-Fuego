# Docker Deployment Backups

> **Snapshot date:** 2026-02-12 (commit `17287a3` on `main`)
> **Status:** Last known working configuration after Alpine→Debian-slim fix and `--reload=false` removal.

## What's here

| Backup file | Restores to | Purpose |
|---|---|---|
| `backend/Dockerfile` | `backend/Dockerfile` | Backend container (Python 3.11-slim, uvicorn, FastAPI) |
| `backend/entrypoint.sh` | `backend/entrypoint.sh` | Alt startup script (import check + uvicorn) |
| `frontend/Dockerfile` | `Dockerfile` (repo root) | Frontend container (Node 20 + nginx, multi-stage) |
| `docker-compose.yml` | `docker-compose.yml` (repo root) | Local dev/prod compose |
| `render.yaml` | `render.yaml` (repo root) | Render service definitions |

## How to restore for deployment

### Quick restore (PowerShell)
```powershell
# From repo root
Copy-Item docker-backups/backend/Dockerfile      backend/Dockerfile      -Force
Copy-Item docker-backups/backend/entrypoint.sh    backend/entrypoint.sh   -Force
Copy-Item docker-backups/frontend/Dockerfile      Dockerfile              -Force
Copy-Item docker-backups/docker-compose.yml       docker-compose.yml      -Force
Copy-Item docker-backups/render.yaml              render.yaml             -Force
```

### Quick restore (bash/Linux)
```bash
# From repo root
cp docker-backups/backend/Dockerfile      backend/Dockerfile
cp docker-backups/backend/entrypoint.sh   backend/entrypoint.sh
cp docker-backups/frontend/Dockerfile     Dockerfile
cp docker-backups/docker-compose.yml      docker-compose.yml
cp docker-backups/render.yaml             render.yaml
```

## Key fixes baked into these backups

1. **Backend base image:** `python:3.11-slim` (Debian) — NOT Alpine. Alpine's musl breaks numpy/scipy/qutip.
2. **No `--reload` flag:** uvicorn `--reload` is dev-only; was removed from production start script.
3. **SDK imports guarded:** `roboto_sai_sdk` is optional via try/except in Python source (not a Docker change, but relevant).
4. **Frontend Render auto-deploy: off** — prevents wasting build minutes during development.

## Important notes

- **Do NOT use `--reload=false`** in uvicorn commands. It's a boolean flag: either `--reload` (on) or omit it (off).
- Backend `autoDeployTrigger: commit` means any push to `main` triggers a Render backend build.
- Frontend `autoDeployTrigger: off` — must trigger manually from Render dashboard.
