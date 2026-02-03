# Filesystem MCP Server Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY mcp-servers/fs-server/package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY mcp-servers/fs-server/src/ ./src/
COPY mcp-servers/fs-server/tsconfig.json ./

# Build TypeScript
RUN npm run build

# Expose port (internal, no need to expose)
# This runs as stdio server

# Start
CMD ["node", "dist/index.js"]