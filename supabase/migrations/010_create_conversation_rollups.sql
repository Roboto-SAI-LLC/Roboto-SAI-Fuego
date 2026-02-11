BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.conversation_rollups (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_topics TEXT[] DEFAULT '{}'::TEXT[],
    sentiment TEXT,
    sentiment_score FLOAT CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    embedding VECTOR(1536),
    covered_until_created_at TIMESTAMPTZ,
    message_count INTEGER DEFAULT 0 CHECK (message_count >= 0),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

DO $$
BEGIN
    IF to_regclass('public.conversation_rollups') IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'conversation_rollups_user_session_unique'
        ) THEN
            ALTER TABLE public.conversation_rollups
                ADD CONSTRAINT conversation_rollups_user_session_unique UNIQUE (user_id, session_id);
        END IF;
    END IF;
END;
$$;

ALTER TABLE public.conversation_rollups ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own rollups" ON public.conversation_rollups;
DROP POLICY IF EXISTS "Users can insert own rollups" ON public.conversation_rollups;
DROP POLICY IF EXISTS "Users can update own rollups" ON public.conversation_rollups;
DROP POLICY IF EXISTS "Users can delete own rollups" ON public.conversation_rollups;
DROP POLICY IF EXISTS "Service role can manage rollups" ON public.conversation_rollups;

CREATE POLICY "Users can view own rollups" ON public.conversation_rollups
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own rollups" ON public.conversation_rollups
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own rollups" ON public.conversation_rollups
    FOR UPDATE USING (auth.uid()::text = user_id);

CREATE POLICY "Users can delete own rollups" ON public.conversation_rollups
    FOR DELETE USING (auth.uid()::text = user_id);

CREATE POLICY "Service role can manage rollups" ON public.conversation_rollups
    FOR ALL USING (auth.role() = 'service_role');

DROP TRIGGER IF EXISTS trigger_update_conversation_rollups_timestamp ON public.conversation_rollups;
CREATE TRIGGER trigger_update_conversation_rollups_timestamp
    BEFORE UPDATE ON public.conversation_rollups
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

GRANT SELECT, INSERT, UPDATE, DELETE ON public.conversation_rollups TO authenticated;
GRANT ALL ON public.conversation_rollups TO service_role;

COMMIT;
