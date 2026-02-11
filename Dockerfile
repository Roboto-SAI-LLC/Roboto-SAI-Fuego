# Roboto SAI 2026 - Fuego Eterno Edition
# Multi-stage Dockerfile with Node.js and Python support
# Created for development flexibility and production optimization

# Stage 1: Base with Python support (for any Node.js Python dependencies)
FROM node:20-alpine AS base

# Install Python and essential build tools for Alpine
RUN apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir uv \
    && ln -sf /opt/venv/bin/uvx /usr/local/bin/uvx

# Ensure virtualenv binaries are on PATH
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies with production flag for smaller image
RUN npm ci --only=production --legacy-peer-deps && npm cache clean --force

# Stage 2: Development
FROM base AS development

# Set development environment
ENV NODE_ENV=development

# Copy application code
COPY . .

# Expose Vite dev server port (from vite.config.ts)
EXPOSE 8080

# Start development server with host binding
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# Stage 2: Build
FROM base AS build

# Accept build arguments for environment variables (will be set by Render)
ARG VITE_API_BASE_URL
ARG VITE_API_URL
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY

# Set environment variables for build (fallback to production URLs)
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL:-https://roboto-sai-backend.onrender.com}
ENV VITE_API_URL=${VITE_API_URL:-https://roboto-sai-backend.onrender.com}
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY
ENV NODE_ENV=production

# Copy application code
COPY . .

# Build the application
RUN npm run build && npm prune --production

# Stage 3: Production (optimized for Render)
FROM nginx:alpine AS production

# Install security updates and curl for health checks
RUN apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create nginx user and directories (skip if already exists)
RUN addgroup -g 1000 -S nginx 2>/dev/null || true && \
    adduser -S -D -H -u 1000 -h /var/cache/nginx -s /sbin/nologin -G nginx -g nginx nginx 2>/dev/null || true

# Copy built files from build stage
COPY --from=build --chown=nginx:nginx /app/dist /usr/share/nginx/html

# Copy optimized nginx configuration for Render
COPY <<'EOF' /etc/nginx/conf.d/default.conf
server {
    listen 10000;  # Render's internal port, will be mapped externally
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Enable gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Health check endpoint for Render
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }

    # Main location block for SPA
    location / {
        try_files $uri $uri/ /index.html;

        # Cache static assets aggressively
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot|webp|avif)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            add_header X-Content-Type-Options nosniff;
            access_log off;
        }

        # Cache HTML with shorter expiry
        location ~* \.html$ {
            expires 1h;
            add_header Cache-Control "public, must-revalidate, proxy-revalidate";
        }
    }

    # Error pages
    error_page 404 /index.html;
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
EOF

# Create health check file
RUN echo "healthy" > /usr/share/nginx/html/health

# Optimize nginx configuration
RUN echo 'worker_processes auto; \
worker_rlimit_nofile 1024; \
events { \
    worker_connections 1024; \
    use epoll; \
    multi_accept on; \
} \
http { \
    include /etc/nginx/mime.types; \
    default_type application/octet-stream; \
    sendfile on; \
    tcp_nopush on; \
    tcp_nodelay on; \
    keepalive_timeout 65; \
    types_hash_max_size 2048; \
    client_max_body_size 16M; \
    server_tokens off; \
}' > /etc/nginx/nginx.conf

# Health check for Render
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Switch to non-root user for security
USER nginx

EXPOSE 10000

CMD ["nginx", "-g", "daemon off;"]
