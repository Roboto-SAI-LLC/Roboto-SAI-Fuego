# OS Agent Dockerfile with MCP servers
FROM node:18-alpine

# Install Python for browser MCP server
RUN apk add --no-cache python3 py3-pip build-base

WORKDIR /app

# Copy and build MCP servers first
COPY mcp-servers/fs-server/package*.json ./fs-server/
COPY mcp-servers/fs-server/src/ ./fs-server/src/
COPY mcp-servers/fs-server/tsconfig.json ./fs-server/
WORKDIR /app/fs-server
RUN npm ci && npm run build

WORKDIR /app
COPY mcp-servers/browser-server/requirements.txt ./browser-server/
COPY mcp-servers/browser-server/server.py ./browser-server/
WORKDIR /app/browser-server
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN npm install -g playwright@1.58.0 && playwright install chromium

# Now OS Agent
WORKDIR /app
COPY os-agent/package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY os-agent/src/ ./src/
COPY os-agent/tsconfig.json ./
COPY os-agent/.env ./

# Build TypeScript
RUN npm run build

# Expose port
EXPOSE 5055

# Start
CMD ["node", "dist/main.js"]