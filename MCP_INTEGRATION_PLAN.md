# Roboto SAI Fuego → Local Sovereign Agent: MCP Integration Plan

**Date**: February 1, 2026  
**Status**: Ready for Implementation  
**Owner**: Roberto Villarreal Martinez

---

## 🎯 Vision

Transform Roboto SAI Fuego into a **local, sovereign agent** that can:
- ✅ Interact with local machine (filesystem, shell, apps)
- ✅ Control browser and search web in real-time  
- ✅ Read/write/edit project files and help with coding
- ✅ Send emails
- ✅ Post on Twitter (and later, create accounts / manage social flows)
- ✅ Consume multiple MCP servers and surface them in UI
- ✅ Expose Roboto SAI's own tools as an MCP server for other hosts

**Key Constraint**: Keep mythic/creative layer in prompts/UI; keep functional layer (tools, MCP, OS access) strictly typed and governed.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (src/)                    │
│  ┌──────────┬──────────────┬────────────────┬─────────────┐ │
│  │ ChatPanel│ToolApproval  │McpServerManager│McpAppFrame  │ │
│  │          │Modal         │                │             │ │
│  └──────────┴──────────────┴────────────────┴─────────────┘ │
│                    ↕ useRobotoClient hook                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │  RobotoClient (SDK)   │
                │  - streamChat()       │
                │  - callBackendTool()  │
                │  - callMcpTool()      │
                │  - approveAction()    │
                └───────┬───────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐  ┌────▼────────┐  ┌──▼────────────┐
│   Backend    │  │  OS Agent   │  │ MCP Servers   │
│ (Python API) │  │  (Daemon)   │  │ (Tools Layer) │
│              │  │             │  │               │
│ - /api/chat  │  │ MCP Host    │  │ - fs-server   │
│ - Tools:     │  │ - Loads MCP │  │ - browser-srv │
│   QIOM,      │  │   servers   │  │ - email-srv   │
│   memory,    │  │ - Permissions│  │ - twitter-srv │
│   cortex     │  │ - HTTP/WS   │  │ - sai-internal│
│              │  │   API       │  │               │
└──────────────┘  └─────────────┘  └───────────────┘
```

---

## 📦 Target Folder Structure

```
roboto-sai-fuego/
├── backend/                      # Existing Python backend
│   ├── app/                      # Core API, routes, tool endpoints
│   └── ...
│
├── sdk/                          # Existing SDK - extend RobotoClient
│   ├── src/
│   │   ├── index.ts              # SDK entrypoint
│   │   ├── robotoClient.ts       # ★ EXTEND: Add MCP support
│   │   └── types.ts              # ★ EXTEND: Add MCP types
│   └── package.json
│
├── src/                          # Vite + React frontend
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── ChatPanel.tsx         # ★ NEW: Tool call visualization
│   │   ├── ToolApprovalModal.tsx # ★ NEW: Approve/deny actions
│   │   ├── McpServerManager.tsx  # ★ NEW: Enable/disable MCP servers
│   │   └── McpAppFrame.tsx       # ★ NEW: Generic MCP app frame
│   ├── hooks/
│   │   └── useRobotoClient.ts    # ★ NEW: Hook to access RobotoClient
│   └── ...
│
├── os-agent/                     # ★ NEW: Local OS agent daemon
│   ├── package.json
│   ├── src/
│   │   ├── index.ts              # Entry: MCP host + servers
│   │   ├── mcpHost.ts            # MCP host implementation
│   │   ├── permissions.ts        # Local permission model
│   │   ├── serverRegistry.ts     # Registry of MCP servers
│   │   └── api.ts                # HTTP/WebSocket API
│   └── README.md
│
├── mcp-servers/                  # ★ NEW: Local MCP servers
│   ├── fs-server/
│   │   ├── package.json
│   │   ├── src/index.ts          # Tools: listDir, readFile, writeFile
│   │   └── mcp.config.json
│   ├── browser-server/
│   │   ├── package.json
│   │   ├── src/index.ts          # Tools: openPage, searchWeb, click
│   │   └── mcp.config.json
│   ├── email-server/
│   │   ├── package.json
│   │   ├── src/index.ts          # Tools: sendEmail, listInbox
│   │   └── mcp.config.json
│   ├── twitter-server/
│   │   ├── package.json
│   │   ├── src/index.ts          # Tools: login, postTweet
│   │   └── mcp.config.json
│   └── sai-internal-server/
│       ├── package.json
│       ├── src/index.ts          # Wrap backend tools as MCP
│       └── mcp.config.json
│
├── scripts/
│   ├── dev-all.sh                # ★ NEW: Start all services
│   └── dev-all.ps1               # ★ NEW: PowerShell version
│
└── README_INTEGRATION.md         # ★ UPDATE: MCP + OS agent usage
```

---

## 🎯 Implementation Order

### Phase 1: SDK Foundation (High Priority)
**Goal**: Extend RobotoClient to support MCP tool routing

1. **Update `sdk/src/types.ts`**:
   ```typescript
   export interface ChatEvent {
     type: "assistant_message" | "tool_call" | "tool_result" | "approval_required" | "error";
     data: any;
   }
   
   export interface ToolCall {
     id: string;
     source: "backend" | "mcp";
     serverId?: string;
     toolName: string;
     args: Record<string, any>;
   }
   
   export interface ApprovalRequest {
     id: string;
     description: string;
     riskLevel: "low" | "medium" | "high";
     suggestedAction: string;
   }
   ```

2. **Extend `sdk/src/robotoClient.ts`**:
   ```typescript
   export class RobotoClient {
     constructor({ 
       backendBaseUrl, 
       osAgentBaseUrl = "http://localhost:5055" 
     }) { ... }
     
     async *streamChat({ sessionId, message }): AsyncGenerator<ChatEvent> { ... }
     async callBackendTool(toolName: string, args: any) { ... }
     async callMcpTool(serverId: string, toolName: string, args: any) { ... }
     async approveAction(actionId: string) { ... }
     async denyAction(actionId: string) { ... }
   }
   ```

3. **Create `sdk/src/index.ts` exports**:
   ```typescript
   export { RobotoClient } from './robotoClient';
   export type { ChatEvent, ToolCall, ApprovalRequest } from './types';
   ```

---

### Phase 2: OS Agent Daemon (High Priority)
**Goal**: Local daemon that hosts MCP servers and enforces permissions

1. **Create `os-agent/package.json`**:
   ```json
   {
     "name": "roboto-os-agent",
     "version": "1.0.0",
     "type": "module",
     "dependencies": {
       "@modelcontextprotocol/sdk": "^1.0.0",
       "express": "^4.18.2",
       "ws": "^8.14.2"
     },
     "scripts": {
       "start": "node src/index.js",
       "dev": "nodemon src/index.js"
     }
   }
   ```

2. **Create `os-agent/src/mcpHost.ts`**:
   - Load server configs from `mcp-servers/*/mcp.config.json`
   - Start each MCP server as child process
   - Maintain connections and tool metadata
   - Provide `callTool(serverId, toolName, args)` method

3. **Create `os-agent/src/permissions.ts`**:
   ```typescript
   export interface PermissionConfig {
     filesystem: {
       read: boolean;
       write: boolean;
       paths: string[];
     };
     browser: { control: boolean };
     email: { send: boolean; allowedDomains: string[] };
     twitter: { post: boolean };
     shell: { allowedCommands: string[] };
   }
   
   export function checkPermission(
     action: string, 
     config: PermissionConfig
   ): { allowed: boolean; reason?: string }
   ```

4. **Create `os-agent/src/api.ts`**:
   ```typescript
   // HTTP endpoints:
   // POST /mcp/call → { serverId, tool, args }
   // POST /approval/decision → { actionId, decision }
   // GET /mcp/servers → list active servers
   ```

---

### Phase 3: MCP Servers (Medium Priority)
**Goal**: Implement core MCP servers for filesystem, browser, etc.

#### 3.1 Filesystem Server (`mcp-servers/fs-server/`)

**Tools**:
- `listDir(path)` → List directory contents
- `readFile(path)` → Read file content
- `writeFile(path, content)` → Write file
- `searchInFiles(rootPath, query)` → Search with grep

**Example `mcp.config.json`**:
```json
{
  "name": "fs-server",
  "description": "Filesystem operations",
  "tools": [
    { "name": "listDir", "description": "List directory", "parameters": {"path": "string"} },
    { "name": "readFile", "description": "Read file", "parameters": {"path": "string"} },
    { "name": "writeFile", "description": "Write file", "parameters": {"path": "string", "content": "string"} }
  ]
}
```

#### 3.2 Browser Server (`mcp-servers/browser-server/`)

**Tools**:
- `openPage(url)` → Navigate to URL
- `searchWeb(query)` → Search search engine
- `click(selector)` → Click element
- `type(selector, text)` → Type text
- `extractContent(selector)` → Extract content

**Tech**: Playwright or Selenium

#### 3.3 Email Server (`mcp-servers/email-server/`)

**Tools**:
- `sendEmail({ to, subject, body })`
- `listInbox(limit?)` → List recent emails
- `readEmail(id)` → Read email content

**Tech**: SMTP/IMAP or Gmail API

#### 3.4 Twitter Server (`mcp-servers/twitter-server/`)

**Tools**:
- `login()` → Authenticate
- `postTweet(text)` → Post tweet
- `getTimeline()` → Get timeline

**Tech**: Twitter API or browser automation via `browser-server`

#### 3.5 SAI Internal Server (`mcp-servers/sai-internal-server/`)

**Tools** (wrapping backend endpoints as MCP):
- `qiom.optimize`
- `memory.query`
- `cortex.oracle.predict`
- `rohub.runWorkflow`

Each tool calls backend HTTP API

---

### Phase 4: Backend Integration (Medium Priority)
**Goal**: Enable backend to propose MCP tool calls in chat responses

1. **Update `backend/api/chat.py`**:
   - Chat endpoint returns `{ type: "tool_call", source: "mcp", serverId, toolName, args }`
   - RobotoClient routes to OS agent

2. **Add `backend/api/tools.py`**:
   - Internal tools with strict JSON schemas
   - Return structured responses

3. **Shared schemas**:
   - Tool input/output definitions
   - Validation with Pydantic

---

### Phase 5: React UI Components (Medium Priority)
**Goal**: Visualize tool calls and handle approvals

1. **Create `src/hooks/useRobotoClient.ts`**:
   ```typescript
   export function useRobotoClient() {
     const client = useMemo(() => new RobotoClient({
       backendBaseUrl: import.meta.env.VITE_BACKEND_URL,
       osAgentBaseUrl: import.meta.env.VITE_OS_AGENT_URL
     }), []);
     
     return { client, sendMessage, events, approveAction, denyAction };
   }
   ```

2. **Create `src/components/chat/ChatPanel.tsx`**:
   - Render chat messages
   - Subscribe to `RobotoClient.streamChat`
   - Show tool calls in UI: "Roboto wants to read file X"
   - Open `ToolApprovalModal` for `approval_required` events

3. **Create `src/components/chat/ToolApprovalModal.tsx`**:
   - Show: description, risk level, suggested action
   - Buttons: Approve / Deny
   - Call `client.approveAction()` or `client.denyAction()`

4. **Create `src/components/mcp/McpServerManager.tsx`**:
   - Fetch list of MCP servers from OS agent: `GET /mcp/servers`
   - Show capabilities (tools, resources)
   - Enable/disable servers
   - Show server status

5. **Create `src/components/mcp/McpAppFrame.tsx`**:
   - Generic container for MCP Apps (if servers expose UI)
   - Show tool metadata and logs

---

### Phase 6: End-to-End Flow Testing (Low Priority)
**Goal**: Validate complete workflow

**Example**: "Read a file and refactor code"

1. User: "Read `src/App.tsx` and suggest improvements"
2. Frontend: `RobotoClient.streamChat({ message })`
3. Backend: Model decides → emit `{ type: "tool_call", source: "mcp", serverId: "fs-server", toolName: "readFile", args: { path: "src/App.tsx" } }`
4. RobotoClient: Sees `source: "mcp"` → POST to OS agent `/mcp/call`
5. OS Agent: Checks permissions → calls `fs-server` via MCP host → returns file content
6. RobotoClient: Emits `{ type: "tool_result", data: "..." }` back to UI and backend
7. Backend: Uses file content → generates refactor suggestions
8. UI: Shows final answer + all tool activity

---

### Phase 7: Permission & Approval Flow (Low Priority)
**Goal**: Secure high-risk actions with user consent

**Example**: "Post on Twitter"

1. User: "Post 'Hello world' on Twitter"
2. Backend: Emit `{ type: "tool_call", source: "mcp", serverId: "twitter-server", toolName: "postTweet" }`
3. OS Agent: Marks as high-risk → emits `{ type: "approval_required", description: "Post tweet", riskLevel: "high" }`
4. UI: Shows `ToolApprovalModal`
5. User: Clicks "Approve"
6. RobotoClient: `client.approveAction(actionId)`
7. OS Agent: Executes tool → returns result
8. UI: Shows success message

---

### Phase 8: Multi-Server Orchestration (Future)
**Goal**: Chain multiple MCP tools in complex workflows

**Example**: "Search web for TypeScript patterns, save to file"

1. `browser-server.searchWeb("TypeScript design patterns")`
2. Extract top 5 results
3. `fs-server.writeFile("patterns.md", content)`
4. Return summary

---

### Permission Model (Scoped Trust Model B)

**Core Philosophy**: Trust but verify with scoped permissions. Enable powerful automation while maintaining user control over sensitive actions.

#### 1. Filesystem Permissions
Roboto SAI is allowed to:
- ✅ **Read & write freely** on development drives:
  - `R:\` (main development)
  - `D:\` (secondary development)
- ✅ **Reads allowed anywhere** (no restrictions)
- ❌ **Write operations outside these drives require approval**
  - Writes outside `R:\` and `D:\` trigger an approval request in the UI

**Effect**: Roboto SAI can code, refactor, create files, and manage projects on main development drives without friction.

#### 2. Browser Automation Permissions
- ✅ **Allowed anywhere** - navigate, click, type, scrape, search, automate any website
- ✅ **No approval required** for normal browsing actions
- ❌ **Account creation restricted** - high-risk actions like creating accounts require explicit approval

**Effect**: Full "Clawbot-style" real-time web control with safety checks on identity creation.

#### 3. Twitter/X Permissions
- ✅ **Posting tweets**: immediate execution (no approval)
  - `postTweet(text)` executes immediately
  - `getTimeline()` executes immediately
- ❌ **Account creation restricted**: Roboto SAI cannot create a Twitter account unless triggered by specific phrases:
  - "Create a Twitter account whenever you want."
  - "Create a Twitter account."
- ⚠️ **Internal decisions**: If the model internally decides it wants to create an account, it must wait for one of these commands.

**Effect**: Frictionless posting but controlled identity creation.

#### 4. Email Permissions
- ❌ **Account creation restricted**: Same rule as Twitter
  - Allowed trigger phrases: "Create an email." / "Create an email account."
- ✅ **Sending emails to user**: Allowed anytime
- ⚠️ **Sending emails to others**: Allowed but must notify user by sending them an email
  - Keeps things interactive and communicative

**Implementation**: When executing email sending, the system must also send a notification email to the user about the action.

#### Permission Enforcement Architecture
- **os-agent/permissions.ts**: Evaluates each tool call against this model
- **Risk Assessment**: Maps actions to risk levels (low/medium/high)
- **Approval Flow**: High-risk actions emit `ApprovalRequired` event to UI
- **Audit Trail**: All permission checks and approvals logged

---

## 🛠️ Development Workflow

### Start All Services

**Bash** (`scripts/dev-all.sh`):
```bash
#!/bin/bash
# Start backend
cd backend && uvicorn main:app --reload --port 5000 &

# Start OS agent
cd os-agent && npm run dev &

# Start frontend
npm run dev &

echo "All services started!"
echo "Backend: http://localhost:5000"
echo "OS Agent: http://localhost:5055"
echo "Frontend: http://localhost:8080"
```

**PowerShell** (`scripts/dev-all.ps1`):
```powershell
# Start backend
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "main:app --reload --port 5000" -WorkingDirectory "backend"

# Start OS agent
Start-Process -NoNewWindow -FilePath "npm" -ArgumentList "run dev" -WorkingDirectory "os-agent"

# Start frontend
Start-Process -NoNewWindow -FilePath "npm" -ArgumentList "run dev"

Write-Host "All services started!"
Write-Host "Backend: http://localhost:5000"
Write-Host "OS Agent: http://localhost:5055"
Write-Host "Frontend: http://localhost:8080"
```

---

## 📊 Success Metrics

- [ ] RobotoClient can route tool calls to backend and OS agent
- [ ] OS agent loads MCP servers and enforces permissions
- [ ] Filesystem server can read/write local files
- [ ] Browser server can search web and extract content
- [ ] Email server can send emails
- [ ] Twitter server can post tweets
- [ ] SAI internal server exposes backend tools
- [ ] React UI shows tool calls and handles approvals
- [ ] End-to-end flow: User request → MCP tools → Response
- [ ] Permission system blocks unauthorized actions
- [ ] Approval flow works for high-risk operations

---

## 🚀 Next Steps

1. **Immediate**: Fix remaining backend errors if any
2. **Phase 1**: Implement RobotoClient SDK extensions (1-2 days)
3. **Phase 2**: Build OS agent daemon (2-3 days)
4. **Phase 3**: Implement fs-server (1 day)
5. **Phase 5**: Add React UI components (2 days)
6. **Phase 6**: End-to-end testing and integration (1 day)

**Total Estimated Effort**: 8-10 days for MVP (Phases 1-6)

---

## 📚 References

- **MCP Protocol**: https://modelcontextprotocol.io/
- **MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **MCP TypeScript SDK**: https://github.com/modelcontextprotocol/typescript-sdk
- **Playwright**: https://playwright.dev/
- **FastAPI**: https://fastapi.tiangolo.com/

---

**Status**: Plan Complete - Ready for Execution  
**Next Action**: Begin Phase 1 (SDK extensions)
