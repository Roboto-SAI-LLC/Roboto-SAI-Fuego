# Roboto SAI OS Agent

MCP (Model Context Protocol) host for Roboto SAI with permission middleware and tool routing.

## Features

- **Scoped Trust Model B**: Permission enforcement for filesystem, browser, Twitter, and email operations
- **MCP Server Management**: Connects to and manages multiple MCP servers
- **Approval Workflows**: High-risk operations require explicit approval
- **REST API**: HTTP endpoints for tool execution and approval management
- **Tool Routing**: Intelligent routing of tool calls to appropriate MCP servers

## Architecture

```
┌─────────────────┐
│  Roboto Backend │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   OS Agent API  │ (Port 5055)
│  permissions.ts │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    mcpHost.ts   │
│  Tool Routing   │
└────────┬────────┘
         │ MCP Protocol
         ▼
┌─────────────────────────────────┐
│      MCP Servers (stdio)        │
│  - filesystem  - browser        │
│  - twitter     - email          │
└─────────────────────────────────┘
```

## Installation

```bash
cd os-agent
npm install
```

## Configuration

Environment variables:

```env
# Server
PORT=5055
HOST=0.0.0.0
LOG_LEVEL=info

# Permission Model
AUTO_APPROVE_LOW_RISK=true

# MCP Servers
ENABLE_TWITTER=false
ENABLE_EMAIL=false

# Twitter (if enabled)
TWITTER_API_KEY=your_key
TWITTER_API_SECRET=your_secret

# Email (if enabled)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email
SMTP_PASS=your_password

# Browser
BROWSER_HEADLESS=true
BROWSER_TIMEOUT=30000

# CORS
ALLOWED_ORIGINS=http://localhost:5000,http://localhost:8080
```

## Usage

### Development

```bash
npm run dev
```

### Production

```bash
npm run build
npm start
```

## API Endpoints

### Health Check
```
GET /health
```

### Get Status
```
GET /api/status
```

### Execute Tool Call
```
POST /api/tools/call
Body: {
  "toolName": "fs_readFile",
  "parameters": { "path": "R:/example.txt" },
  "userId": "user-123",
  "sessionId": "session-456"
}
```

### Get Pending Approvals
```
GET /api/approvals/pending
```

### Approve/Reject Tool Call
```
POST /api/approvals/action
Body: {
  "approvalId": "approval_xxx",
  "action": "approve",
  "userId": "user-123"
}
```

### Get MCP Servers
```
GET /api/servers
```

### Check Permission
```
POST /api/permissions/check
Body: {
  "toolName": "twitter_create_account",
  "parameters": { "username": "roboto_sai" }
}
```

## Scoped Trust Model B

### Filesystem
- **Allowed Drives**: R:/, D:/
- **Read**: ✅ Allowed (LOW risk)
- **Write**: ✅ Allowed (MEDIUM risk)

### Browser
- **Full Automation**: ✅ Allowed
- **Reading/Scrolling**: LOW risk
- **Posting/Login**: MEDIUM risk
- **Purchases/Payments**: HIGH risk, requires approval

### Twitter
- **Reading/Scrolling**: ✅ Allowed (LOW risk)
- **Posting**: ✅ Allowed (MEDIUM risk)
- **Account Creation**: ❌ Requires approval (HIGH risk)

### Email
- **Sending**: ✅ Allowed with notification (MEDIUM risk)

## MCP Servers

MCP servers must be implemented separately in `backend/mcp_servers/`:
- `fs_server.py` - Filesystem operations
- `browser_server.py` - Browser automation
- `twitter_server.py` - Twitter integration
- `email_server.py` - Email sending

## License

MIT
