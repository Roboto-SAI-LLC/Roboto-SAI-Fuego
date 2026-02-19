# Roboto SAI 2026 - Production Endpoints Documentation

## Overview
This document describes the production-ready API endpoints for conversation management, including enhanced history retrieval, session management, and automatic summarization.

**Base URL:** `http://localhost:8080` (development) | `https://roboto-sai.com` (production)

**Authentication:** All endpoints require authentication via session cookie (`roboto_session`) or JWT token.

---

## 1. Enhanced Chat History (GET /api/chat/history)

Retrieve chat messages with cursor-based pagination and advanced filtering.

### Endpoint
```
GET /api/chat/history
```

### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | - | Filter by specific session |
| `role` | string | No | - | Filter by role (`user`, `roboto`, `assistant`) |
| `since` | string | No | - | ISO 8601 timestamp (messages after this time) |
| `until` | string | No | - | ISO 8601 timestamp (messages before this time) |
| `limit` | integer | No | 50 | Max messages per page (1-200) |
| `cursor` | string | No | - | Pagination cursor from previous response |

### Response
```json
{
  "success": true,
  "count": 10,
  "has_more": true,
  "next_cursor": "MjAyNi0wMi0xMVQxMjozNDo1NlosMTIzNDU2Nzg5MA==",
  "messages": [
    {
      "id": "uuid",
      "user_id": "uuid",
      "session_id": "session-123",
      "role": "user",
      "content": "Hello Roboto!",
      "emotion": "excited",
      "emotion_text": "Excited and hopeful",
      "emotion_probabilities": {"joy": 0.8, "anticipation": 0.6},
      "created_at": "2026-02-11T12:34:56Z"
    }
  ],
  "timestamp": "2026-02-11T12:34:56Z"
}
```

### Examples

**Basic retrieval (most recent 50 messages):**
```bash
curl -X GET "http://localhost:8080/api/chat/history?limit=50" \
  -H "Cookie: roboto_session=your_session_token"
```

**Paginated retrieval (next page):**
```bash
curl -X GET "http://localhost:8080/api/chat/history?limit=50&cursor=MjAyNi0wMi0xMVQxMjozNDo1NlosMTIzNDU2Nzg5MA==" \
  -H "Cookie: roboto_session=your_session_token"
```

**Filter by session:**
```bash
curl -X GET "http://localhost:8080/api/chat/history?session_id=qa-session-001&limit=20" \
  -H "Cookie: roboto_session=your_session_token"
```

**Filter by role:**
```bash
curl -X GET "http://localhost:8080/api/chat/history?role=roboto&limit=10" \
  -H "Cookie: roboto_session=your_session_token"
```

**Filter by time range:**
```bash
curl -X GET "http://localhost:8080/api/chat/history?since=2026-02-11T00:00:00Z&until=2026-02-11T23:59:59Z" \
  -H "Cookie: roboto_session=your_session_token"
```

**Combined filters:**
```bash
curl -X GET "http://localhost:8080/api/chat/history?session_id=test-123&role=user&since=2026-02-11T00:00:00Z&limit=25" \
  -H "Cookie: roboto_session=your_session_token"
```

### Performance Notes
- Uses keyset pagination (efficient for large datasets)
- Indexed on `(user_id, session_id, created_at DESC, id DESC)`
- No caching (to ensure fresh data)
- Rate limit: 60 requests/minute

---

## 2. List Sessions (GET /api/sessions)

Retrieve all unique conversation sessions for the authenticated user.

### Endpoint
```
GET /api/sessions
```

### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 50 | Max sessions per page |
| `offset` | integer | No | 0 | Pagination offset |

### Response
```json
{
  "success": true,
  "count": 5,
  "total": 12,
  "sessions": [
    {
      "session_id": "qa-session-001",
      "last_message_time": "2026-02-11T15:30:00Z",
      "message_count": 42,
      "summary_preview": "Discussion about quantum computing applications..."
    }
  ],
  "timestamp": "2026-02-11T15:45:00Z"
}
```

### Examples

**List all sessions:**
```bash
curl -X GET "http://localhost:8080/api/sessions?limit=20" \
  -H "Cookie: roboto_session=your_session_token"
```

**Paginated sessions:**
```bash
curl -X GET "http://localhost:8080/api/sessions?limit=10&offset=10" \
  -H "Cookie: roboto_session=your_session_token"
```

### Fields Description
- `session_id`: Unique session identifier
- `last_message_time`: Timestamp of most recent message
- `message_count`: Total messages in session
- `summary_preview`: First 200 chars of rollup summary (if available)

### Performance Notes
- Aggregates data from `messages` table
- Joins with `conversation_rollups` for summary previews
- Ordered by `last_message_time DESC`
- Rate limit: 60 requests/minute

---

## 3. Get Conversation Rollup (GET /api/conversations/rollup)

Retrieve the current conversation summary/rollup for a specific session.

### Endpoint
```
GET /api/conversations/rollup
```

### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session to retrieve rollup for |

### Response
```json
{
  "success": true,
  "rollup": {
    "user_id": "uuid",
    "session_id": "qa-session-001",
    "summary": "The conversation covered quantum computing applications, focusing on error correction techniques and practical implementations. User expressed interest in learning more about topological qubits.",
    "key_topics": ["quantum computing", "error correction", "topological qubits"],
    "sentiment": "positive",
    "sentiment_score": 0.75,
    "covered_until_created_at": "2026-02-11T15:30:00Z",
    "message_count": 42,
    "updated_at": "2026-02-11T15:35:00Z"
  },
  "timestamp": "2026-02-11T15:45:00Z"
}
```

### Error Responses
**404 Not Found** - No rollup exists for the session:
```json
{
  "detail": "No rollup found for session: qa-session-001"
}
```

### Examples

**Get rollup for a session:**
```bash
curl -X GET "http://localhost:8080/api/conversations/rollup?session_id=qa-session-001" \
  -H "Cookie: roboto_session=your_session_token"
```

### Fields Description
- `summary`: AI-generated conversation summary
- `key_topics`: Extracted key topics (array of strings)
- `sentiment`: Overall sentiment (`positive`, `negative`, `neutral`, `mixed`)
- `sentiment_score`: Numeric sentiment score (-1 to 1)
- `covered_until_created_at`: Timestamp of last message included in rollup
- `message_count`: Number of messages summarized
- `updated_at`: When rollup was last updated

### Performance Notes
- Direct lookup in `conversation_rollups` table
- Unique constraint on `(user_id, session_id)`
- Rate limit: 60 requests/minute

---

## 4. Create/Update Conversation Summary (POST /api/conversations/summarize)

Manually trigger conversation summarization for a session. Uses Grok to generate intelligent summaries.

### Endpoint
```
POST /api/conversations/summarize
```

### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session to summarize |
| `message_limit` | integer | No | 50 | Max messages to include (1-200) |
| `force` | boolean | No | false | Force re-summarization even if recent |

### Response
```json
{
  "success": true,
  "rollup": {
    "user_id": "uuid",
    "session_id": "qa-session-001",
    "summary": "The conversation explored quantum computing fundamentals...",
    "key_topics": ["quantum computing", "qubits", "superposition"],
    "sentiment": "positive",
    "sentiment_score": 0.8,
    "covered_until_created_at": "2026-02-11T15:30:00Z",
    "message_count": 42,
    "updated_at": "2026-02-11T15:45:00Z"
  },
  "messages_summarized": 42,
  "timestamp": "2026-02-11T15:45:00Z"
}
```

### Conditional Response
If rollup is recent and `force=false`:
```json
{
  "success": true,
  "message": "Rollup is recent, skipping. Use force=true to override.",
  "rollup": { ... },
  "timestamp": "2026-02-11T15:45:00Z"
}
```

### Examples

**Create summary:**
```bash
curl -X POST "http://localhost:8080/api/conversations/summarize?session_id=qa-session-001" \
  -H "Cookie: roboto_session=your_session_token"
```

**Force re-summarization:**
```bash
curl -X POST "http://localhost:8080/api/conversations/summarize?session_id=qa-session-001&force=true" \
  -H "Cookie: roboto_session=your_session_token"
```

**Summarize specific message count:**
```bash
curl -X POST "http://localhost:8080/api/conversations/summarize?session_id=qa-session-001&message_limit=100" \
  -H "Cookie: roboto_session=your_session_token"
```

### AI Summarization
- Uses Grok 4.1 (grok-4-1-fast-reasoning)
- Extracts: summary, key topics, sentiment, entities
- Semantic embeddings stored for search (future feature)
- Upserts to `conversation_rollups` table

### Performance Notes
- Rate limit: 20 requests/minute (Grok API intensive)
- Skips if updated within last 10 minutes (override with `force=true`)
- Processes up to 200 messages
- Background execution possible

---

## 5. Auto-Summarization

### Background Process
Conversation summarization is **automatically triggered** after:
1. **20 new messages** since last rollup
2. **OR 10 minutes of continuous activity**

### Implementation
Auto-summarization is integrated into `POST /api/chat`:
```python
# After saving messages
await _maybe_trigger_auto_summarization(user_id, session_id, background_tasks)
```

### Monitoring
Check server logs for:
```
Auto-triggering summarization for session qa-session-001 (20+ new messages)
Background summarization completed for session qa-session-001
```

### Configuration
Thresholds can be adjusted in `_maybe_trigger_auto_summarization()`:
- `new_message_count >= 20` → Trigger
- `activity_duration >= 600` seconds (10 min) → Trigger

---

## Architecture Decisions

### 1. Keyset Pagination (vs Offset Pagination)
**Why:** Efficient for large datasets, no performance degradation with deep pages.

**Index:** `idx_messages_user_session_pagination (user_id, session_id, created_at DESC, id DESC)`

**Cursor encoding:** Base64(`"created_at,id"`)

### 2. Conversation Rollups (vs Conversation Summaries)
**Why:** Distinction between legacy summaries and new semantic rollups.

**Table:** `conversation_rollups` includes:
- Semantic embeddings (`vector(1536)`)
- Covered timestamp tracking
- Search function: `search_conversation_rollups()`

### 3. Incremental Summarization
**Why:** Reduces Grok API calls, prevents duplicate summaries.

**Logic:** Only summarize new messages since `covered_until_created_at`.

### 4. Background Execution
**Why:** Don't block chat responses for summarization.

**Implementation:** FastAPI `BackgroundTasks` for async summarization.

---

## Database Schema

### Messages Table
```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'roboto', 'assistant')),
    content TEXT NOT NULL,
    emotion TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_messages_user_session_pagination
    ON messages (user_id, session_id, created_at DESC, id DESC);
```

### Conversation Rollups Table
```sql
CREATE TABLE conversation_rollups (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_topics TEXT[] DEFAULT '{}',
    sentiment TEXT,
    sentiment_score FLOAT CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    embedding VECTOR(1536),
    covered_until_created_at TIMESTAMPTZ,
    message_count INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, session_id)
);

CREATE INDEX idx_conv_rollups_embedding
    ON conversation_rollups USING ivfflat (embedding vector_cosine_ops);
```

---

## Error Handling

### Common Error Codes
| Code | Meaning | Example |
|------|---------|---------|
| 400 | Bad Request | Invalid cursor format, invalid role |
| 401 | Unauthorized | Missing or expired session token |
| 404 | Not Found | No rollup exists for session |
| 429 | Too Many Requests | Rate limit exceeded |
| 503 | Service Unavailable | Supabase not configured |

### Example Error Response
```json
{
  "detail": "Invalid cursor format"
}
```

---

## Rate Limits

| Endpoint | Limit | Scope |
|----------|-------|-------|
| GET /api/chat/history | 60/min | Per user |
| GET /api/sessions | 60/min | Per user |
| GET /api/conversations/rollup | 60/min | Per user |
| POST /api/conversations/summarize | 20/min | Per user |
| POST /api/chat | 30/min | Per user |

---

## Migration & Deployment

### Prerequisites
1. Supabase database with migrations applied:
   - `010_create_conversation_rollups.sql`
   - `011_add_rollups_indexes_and_search.sql`

2. Environment variables:
   ```bash
   XAI_API_KEY=your_grok_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

### Migration Steps
1. Apply database migrations (Supabase)
2. Deploy updated backend code
3. Verify endpoints with test script:
   ```bash
   python test_production_endpoints.py
   ```

### Rollback Plan
If issues occur:
1. No breaking changes to existing `/api/chat` endpoint
2. New endpoints can be disabled via feature flags
3. Database rollups table is isolated (no impact on messages)

---

## Frontend Integration

### Example: Paginated History Component
```typescript
async function loadChatHistory(sessionId: string, cursor?: string) {
  const params = new URLSearchParams({
    session_id: sessionId,
    limit: '50',
    ...(cursor && { cursor })
  });
  
  const response = await fetch(`/api/chat/history?${params}`, {
    credentials: 'include'
  });
  
  const data = await response.json();
  return {
    messages: data.messages,
    nextCursor: data.next_cursor,
    hasMore: data.has_more
  };
}
```

### Example: Session List Component
```typescript
async function loadSessions() {
  const response = await fetch('/api/sessions?limit=20', {
    credentials: 'include'
  });
  
  const data = await response.json();
  return data.sessions; // { session_id, last_message_time, message_count, summary_preview }
}
```

### Example: Display Rollup
```typescript
async function loadConversationSummary(sessionId: string) {
  const response = await fetch(`/api/conversations/rollup?session_id=${sessionId}`, {
    credentials: 'include'
  });
  
  if (response.status === 404) {
    return null; // No rollup yet
  }
  
  const data = await response.json();
  return data.rollup; // { summary, key_topics, sentiment, ... }
}
```

---

## Testing

### Run Test Suite
```bash
python test_production_endpoints.py
```

### Manual Testing (curl)
See examples in each endpoint section above.

### Expected Results
- ✅ Chat history pagination works
- ✅ Sessions listed with metadata
- ✅ Rollups retrievable after summarization
- ✅ Auto-summarization triggers in background
- ✅ No breaking changes to existing chat flow

---

## Performance Considerations

### Database Indexes
- **Messages:** Composite index on `(user_id, session_id, created_at DESC, id DESC)` enables efficient keyset pagination
- **Rollups:** IVFFLAT index on `embedding` for future semantic search

### Caching Strategy
- History: No caching (real-time data priority)
- Sessions: No caching (frequently updated)
- Rollups: Consider caching (changes less frequently)

### Optimization Tips
1. **Limit page size:** Use `limit=50` or less for responsive UIs
2. **Filter early:** Apply `session_id` filter to reduce dataset
3. **Background summarization:** Don't block UI for summaries
4. **Monitor rate limits:** Respect 20/min for summarization endpoint

---

## Future Enhancements

1. **Semantic Search:** Use `search_conversation_rollups()` function for AI-powered search
2. **Real-time Updates:** WebSocket support for live history updates
3. **Export:** Download full conversation history (CSV/JSON)
4. **Analytics:** Conversation metrics dashboard (topics, sentiment trends)
5. **Multi-language:** Summarization in user's preferred language

---

## Support

For issues or questions:
- GitHub Issues: [roboto-sai-2026/issues](https://github.com/your-org/roboto-sai-2026/issues)
- Documentation: [/docs](https://docs.yoursite.com)
- API Status: [/api/health](http://localhost:8080/api/health)

---

**Version:** 1.0.0  
**Last Updated:** February 11, 2026  
**Author:** Roberto Villarreal Martinez
