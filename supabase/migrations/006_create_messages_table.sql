-- Create messages table for chat history
-- This table stores all chat messages between users and Roboto SAI

BEGIN;

-- Create messages table
CREATE TABLE IF NOT EXISTS public.messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'roboto', 'assistant')),
    content TEXT NOT NULL,
    emotion TEXT,
    emotion_text TEXT,
    emotion_probabilities JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON public.messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON public.messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON public.messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_user_session ON public.messages(user_id, session_id, created_at);

-- Create message_feedback table
CREATE TABLE IF NOT EXISTS public.message_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID NOT NULL REFERENCES public.messages(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating IN (1, -1)),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(message_id, user_id)
);

-- Create indexes for feedback
CREATE INDEX IF NOT EXISTS idx_message_feedback_message_id ON public.message_feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_message_feedback_user_id ON public.message_feedback(user_id);

-- Enable RLS
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.message_feedback ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can view own messages" ON public.messages;
DROP POLICY IF EXISTS "Users can insert own messages" ON public.messages;
DROP POLICY IF EXISTS "Users can update own messages" ON public.messages;
DROP POLICY IF EXISTS "Service role can manage messages" ON public.messages;

DROP POLICY IF EXISTS "Users can view feedback on own messages" ON public.message_feedback;
DROP POLICY IF EXISTS "Users can insert feedback on accessible messages" ON public.message_feedback;
DROP POLICY IF EXISTS "Users can update own feedback" ON public.message_feedback;
DROP POLICY IF EXISTS "Users can delete own feedback" ON public.message_feedback;
DROP POLICY IF EXISTS "Service role can manage feedback" ON public.message_feedback;

-- Create RLS policies for messages
CREATE POLICY "Users can view own messages" ON public.messages
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own messages" ON public.messages
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own messages" ON public.messages
    FOR UPDATE USING (auth.uid() = user_id);

-- Service role can manage all messages (for backend operations)
CREATE POLICY "Service role can manage messages" ON public.messages
    FOR ALL USING (auth.role() = 'service_role');

-- Create RLS policies for message_feedback
CREATE POLICY "Users can view feedback on own messages" ON public.message_feedback
    FOR SELECT USING (
        auth.uid() = user_id OR
        EXISTS (
            SELECT 1 FROM public.messages
            WHERE messages.id = message_feedback.message_id
            AND messages.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert feedback on accessible messages" ON public.message_feedback
    FOR INSERT WITH CHECK (
        auth.uid() = user_id AND
        EXISTS (
            SELECT 1 FROM public.messages
            WHERE messages.id = message_feedback.message_id
        )
    );

CREATE POLICY "Users can update own feedback" ON public.message_feedback
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own feedback" ON public.message_feedback
    FOR DELETE USING (auth.uid() = user_id);

-- Service role can manage all feedback
CREATE POLICY "Service role can manage feedback" ON public.message_feedback
    FOR ALL USING (auth.role() = 'service_role');

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON public.messages TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.message_feedback TO authenticated;
GRANT ALL ON public.messages TO service_role;
GRANT ALL ON public.message_feedback TO service_role;

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_messages_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-updating updated_at
CREATE TRIGGER trigger_update_messages_timestamp
    BEFORE UPDATE ON public.messages
    FOR EACH ROW
    EXECUTE FUNCTION update_messages_updated_at();

COMMIT;
