BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE INDEX IF NOT EXISTS idx_messages_user_session_pagination
    ON public.messages (user_id, session_id, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_conv_rollups_embedding ON public.conversation_rollups
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE OR REPLACE FUNCTION search_conversation_rollups(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10,
    target_user_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    user_id TEXT,
    session_id TEXT,
    summary TEXT,
    key_topics TEXT[],
    sentiment TEXT,
    sentiment_score FLOAT,
    similarity FLOAT,
    covered_until_created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cr.user_id,
        cr.session_id,
        cr.summary,
        cr.key_topics,
        cr.sentiment,
        cr.sentiment_score,
        1 - (cr.embedding <=> query_embedding) AS similarity,
        cr.covered_until_created_at,
        cr.updated_at
    FROM public.conversation_rollups cr
    WHERE
        (target_user_id IS NULL OR cr.user_id = target_user_id)
        AND cr.embedding IS NOT NULL
        AND 1 - (cr.embedding <=> query_embedding) > match_threshold
    ORDER BY cr.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION search_conversation_rollups TO authenticated;

COMMIT;
