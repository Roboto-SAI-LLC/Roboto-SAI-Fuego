/**
 * Roboto SAI Chat Page
 * Created by Roberto Villarreal Martinez for Roboto SAI (powered by Grok)
 * The heart of the empire - where fire meets conversation
 * Connected to FastAPI backend with xAI Grok integration
 */

import { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { formatDistanceToNow } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStore, FileAttachment, Message } from '@/stores/chatStore';
import { useMemoryStore } from '@/stores/memoryStore';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { ChatInput } from '@/components/chat/ChatInput';
import { TypingIndicator } from '@/components/chat/TypingIndicator';
import { EmberParticles } from '@/components/effects/EmberParticles';
import { Header } from '@/components/layout/Header';
import { ChatSidebar, ChatSidebarSessionItem } from '@/components/chat/ChatSidebar';
import { VoiceMode } from '@/components/chat/VoiceMode';
import { Flame, Loader2, RefreshCw, Skull, Sparkles, MessageSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { useToast } from '@/components/ui/use-toast';
import { useRobotoClient } from '@/hooks/useRobotoClient';
import {
  ChatHistoryPage,
  ChatRollupSummary,
  normalizeHistoryResponse,
  normalizeRollupResponse,
  normalizeSessionsResponse,
  parseTimestamp,
} from '@/lib/chatApi';

type ApiError = Error & { status?: number };
type SidebarSession = ChatSidebarSessionItem & { last_message_at?: string | number };

const HISTORY_PAGE_SIZE = 40;

const buildTitle = (content: string): string => {
  const cleaned = content.replace(/[^\w\s]/g, '').trim();
  const words = cleaned.split(/\s+/).slice(0, 5);
  return words.join(' ') || 'New Chat';
};

const formatSessionTime = (value?: string | number): string => {
  if (!value) return 'New';
  const parsed = parseTimestamp(value);
  if (Number.isNaN(parsed.getTime())) return 'Unknown';
  return formatDistanceToNow(parsed, { addSuffix: true });
};

const Chat = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { userId, isLoggedIn, username, email } = useAuthStore();
  const {
    getMessages,
    isLoading,
    ventMode,
    voiceMode,
    currentTheme,
    addMessage,
    setLoading,
    toggleVentMode,
    toggleVoiceMode,
    getAllConversationsContext,
    createNewConversation,
    selectConversation,
    currentConversationId,
  } = useChatStore();

  const { buildContextForAI, addMemory, trackEntity, isReady: memoryReady } = useMemoryStore();
  const {
    sendMessage: streamMessage,
  } = useRobotoClient();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [localSessions, setLocalSessions] = useState<SidebarSession[]>([]);

  const rainDrops = useMemo(
    () =>
      Array.from({ length: 20 }).map(() => {
        const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
        return {
          id,
          left: `${Math.random() * 100}%`,
          height: `${Math.random() * 100 + 50}px`,
          duration: Math.random() * 2 + 1,
          delay: Math.random() * 2,
        };
      }),
    []
  );

  const apiBaseUrl = useMemo(() => {
    const envUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '';
    const fallbackBase = globalThis.window?.location.origin ?? '';
    return (envUrl || fallbackBase).replace(/\/+$/, '').replace(/\/api$/, '');
  }, []);

  const apiFetch = useCallback(async <T,>(path: string, init?: RequestInit): Promise<T> => {
    const response = await fetch(`${apiBaseUrl}${path}`, {
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers || {}),
      },
      ...init,
    });

    if (response.status === 401) {
      navigate('/login');
      const unauthorizedError = new Error('Unauthorized') as ApiError;
      unauthorizedError.status = 401;
      throw unauthorizedError;
    }

    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        const data = await response.json() as Record<string, unknown>;
        message = (typeof data.detail === 'string' && data.detail)
          || (typeof data.error === 'string' && data.error)
          || message;
      } catch {
        // ignore JSON parsing errors
      }
      const error = new Error(message) as ApiError;
      error.status = response.status;
      throw error;
    }

    return response.json() as Promise<T>;
  }, [apiBaseUrl, navigate]);

  const sessionsQuery = useQuery({
    queryKey: ['chat-sessions'],
    queryFn: async () => {
      const data = await apiFetch<unknown>('/api/sessions');
      return normalizeSessionsResponse(data).map((session) => ({
        id: session.id,
        title: session.title,
        messageCount: session.message_count,
        preview: session.summary_preview || session.preview,
        last_message_at: session.last_message_at,
      }));
    },
    staleTime: 10000,
    retry: 1,
  });

  const mergedSessions = useMemo(() => {
    const sessionMap = new Map<string, SidebarSession>();
    (sessionsQuery.data ?? []).forEach((session) => {
      sessionMap.set(session.id, session);
    });

    localSessions.forEach((session) => {
      if (!sessionMap.has(session.id)) {
        sessionMap.set(session.id, session);
      } else {
        const existing = sessionMap.get(session.id);
        sessionMap.set(session.id, {
          ...(existing || session),
          ...session,
        });
      }
    });

    return Array.from(sessionMap.values()).sort((a, b) => {
      const timeA = a.last_message_at ? parseTimestamp(a.last_message_at).getTime() : 0;
      const timeB = b.last_message_at ? parseTimestamp(b.last_message_at).getTime() : 0;
      return timeB - timeA;
    });
  }, [localSessions, sessionsQuery.data]);

  useEffect(() => {
    if (!activeSessionId && mergedSessions[0]?.id) {
      setActiveSessionId(mergedSessions[0].id);
      selectConversation(mergedSessions[0].id);
    }
  }, [activeSessionId, mergedSessions, selectConversation]);

  useEffect(() => {
    if (activeSessionId && activeSessionId !== currentConversationId) {
      selectConversation(activeSessionId);
    }
  }, [activeSessionId, currentConversationId, selectConversation]);

  const historyQuery = useInfiniteQuery<
    ChatHistoryPage,
    ApiError,
    ChatHistoryPage,
    [string, string | null],
    string | undefined
  >({
    queryKey: ['chat-history', activeSessionId],
    enabled: Boolean(activeSessionId),
    initialPageParam: undefined,
    queryFn: async ({ pageParam }) => {
      if (!activeSessionId) {
        return { messages: [], hasMore: false, nextCursor: null };
      }

      const params = new URLSearchParams();
      params.set('limit', String(HISTORY_PAGE_SIZE));
      params.set('session_id', activeSessionId);
      if (pageParam) params.set('cursor', pageParam);

      try {
        const primaryData = await apiFetch<unknown>(`/api/chat/history?${params.toString()}`);
        return normalizeHistoryResponse(primaryData);
      } catch (error) {
        const typedError = error as ApiError;
        if (typedError.status === 404) {
          const fallbackData = await apiFetch<unknown>(`/api/chat/history/paginated?${params.toString()}`);
          return normalizeHistoryResponse(fallbackData);
        }
        throw typedError;
      }
    },
    getNextPageParam: (lastPage) => {
      if (!lastPage.hasMore) return undefined;
      return lastPage.nextCursor || undefined;
    },
    retry: 1,
  });

  const historyMessages = useMemo(() => {
    const pages = historyQuery.data?.pages ?? [];
    const combined = pages.flatMap((page) => page.messages || []);
    return combined.sort((a, b) => parseTimestamp(a.timestamp).getTime() - parseTimestamp(b.timestamp).getTime());
  }, [historyQuery.data]);

  const liveMessages = getMessages();

  const mergedMessages = useMemo(() => {
    const messageMap = new Map<string, Message>();
    [...historyMessages, ...liveMessages].forEach((message) => {
      messageMap.set(message.id, {
        ...message,
        timestamp: parseTimestamp(message.timestamp),
      });
    });

    return Array.from(messageMap.values()).sort(
      (a, b) => parseTimestamp(a.timestamp).getTime() - parseTimestamp(b.timestamp).getTime()
    );
  }, [historyMessages, liveMessages]);

  const summaryQuery = useQuery<ChatRollupSummary, ApiError>({
    queryKey: ['chat-rollup', activeSessionId],
    enabled: Boolean(activeSessionId),
    queryFn: async () => {
      if (!activeSessionId) {
        return { summary: '', topics: [], sentiment: 'neutral', message_count: 0 };
      }
      const data = await apiFetch<unknown>(`/api/conversations/rollup?session_id=${encodeURIComponent(activeSessionId)}`);
      return normalizeRollupResponse(data);
    },
    retry: 1,
  });

  const summaryMutation = useMutation<unknown, ApiError>({
    mutationFn: async () => {
      if (!activeSessionId) return null;
      return apiFetch<unknown>(`/api/conversations/summarize?session_id=${encodeURIComponent(activeSessionId)}`, {
        method: 'POST',
      });
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['chat-rollup', activeSessionId] });
      await queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      toast({
        title: 'Summary refreshed',
        description: 'Conversation rollup updated.',
      });
    },
    onError: (error) => {
      toast({
        variant: 'destructive',
        title: 'Summary failed',
        description: error.message || 'Unable to regenerate summary.',
      });
    },
  });

  const upsertLocalSession = useCallback((sessionId: string, patch: Partial<SidebarSession>) => {
    setLocalSessions((previous) => {
      const existing = previous.find((session) => session.id === sessionId);
      if (!existing) {
        return [{ id: sessionId, title: 'New Chat', ...patch }, ...previous];
      }
      return previous.map((session) => (
        session.id === sessionId
          ? { ...session, ...patch }
          : session
      ));
    });
  }, []);

  const handleSelectSession = useCallback((sessionId: string) => {
    setActiveSessionId(sessionId);
    selectConversation(sessionId);
  }, [selectConversation]);

  const handleNewSession = useCallback(() => {
    const newSessionId = createNewConversation();
    setActiveSessionId(newSessionId);
    upsertLocalSession(newSessionId, {
      title: 'New Chat',
      preview: 'No messages yet',
      messageCount: 0,
      last_message_at: new Date().toISOString(),
    });
  }, [createNewConversation, upsertLocalSession]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [mergedMessages, isLoading, scrollToBottom]);

  // Extract and store important information from conversations
  const extractMemories = async (userMessage: string, robotoResponse: string, sessionId: string) => {
    // Extract potential entities (simple pattern matching - can be enhanced)
    const namePattern = /(?:my (?:name is|friend|brother|sister|mom|dad|wife|husband|partner|boss|colleague) (?:is )?|I'm |I am )([A-Z][a-z]+)/gi;
    let match;
    while ((match = namePattern.exec(userMessage)) !== null) {
      const entityName = match[1];
      const entityType = userMessage.toLowerCase().includes('name is') ? 'self' : 'person';
      await trackEntity(entityName, entityType, userMessage);
    }

    // Detect preferences (simple heuristics)
    const preferencePatterns = [
      { pattern: /I (?:really )?(?:love|like|prefer|enjoy) (.+?)(?:\.|,|!|$)/i, type: 'likes' },
      { pattern: /I (?:hate|dislike|don't like|can't stand) (.+?)(?:\.|,|!|$)/i, type: 'dislikes' },
      { pattern: /I'm (?:a|an) (.+?)(?:\.|,|!|$)/i, type: 'identity' },
    ];

    for (const { pattern, type } of preferencePatterns) {
      const prefMatch = userMessage.match(pattern);
      if (prefMatch) {
        await addMemory(
          `User ${type}: ${prefMatch[1]}`,
          'preferences',
          1.2,
          { source: sessionId, extractedFrom: userMessage }
        );
      }
    }

    void robotoResponse;
  };

  const displayChatError = (errorMessage: string) => {
    let title = "Connection Error";
    let description = "The eternal fire flickers but does not die. Please try again.";

    if (errorMessage.includes('404') || errorMessage.includes('Not Found')) {
      title = "API Endpoint Not Found";
      description = "The chat endpoint is not available. Grok API may be unavailable. Check your deployment configuration.";
    } else if (errorMessage.includes('401') || errorMessage.includes('Unauthorized')) {
      title = "Authentication Required";
      description = "Your session has expired. Please log in again.";
      setTimeout(() => navigate('/login'), 2000);
    } else if (errorMessage.includes('403') || errorMessage.includes('Forbidden')) {
      title = "Access Denied";
      description = "You don't have permission to access this resource.";
    } else if (errorMessage.includes('503') || errorMessage.includes('Service Unavailable')) {
      title = "Service Temporarily Unavailable";
      description = "Grok API is currently unavailable. This may be due to rate limits or API access issues. Try again in a moment.";
    } else if (errorMessage.includes('timeout') || errorMessage.includes('ETIMEDOUT')) {
      title = "Request Timeout";
      description = "The request took too long. Please check your internet connection and try again.";
    } else if (errorMessage.includes('network') || errorMessage.includes('Failed to fetch')) {
      title = "Network Error";
      description = "Cannot connect to the server. Please check your internet connection.";
    } else if (errorMessage.includes('Could not connect to Grok API')) {
      title = "Grok API Unavailable";
      description = "The AI service is currently unavailable. The backend may need configuration or Grok API access.";
    } else if (errorMessage.length > 0 && errorMessage !== 'Connection error') {
      description = errorMessage;
    }

    toast({
      variant: "destructive",
      title,
      description,
    });
  };

  const handleSend = async (content: string, attachments?: FileAttachment[]) => {
    if (!isLoggedIn || !userId) {
      navigate('/login');
      return;
    }

    let sessionId = activeSessionId || currentConversationId;
    if (!sessionId) {
      sessionId = createNewConversation();
    }

    if (sessionId !== currentConversationId) {
      selectConversation(sessionId);
    }
    if (sessionId !== activeSessionId) {
      setActiveSessionId(sessionId);
    }

    const existingSession = mergedSessions.find((session) => session.id === sessionId);

    setLoading(true);

    addMessage({ role: 'user', content, attachments });
    upsertLocalSession(sessionId, {
      title: existingSession?.title || buildTitle(content),
      preview: content,
      messageCount: (existingSession?.messageCount || 0) + 1,
      last_message_at: new Date().toISOString(),
    });

    const conversationContext = getAllConversationsContext();
    const memoryContext = memoryReady ? buildContextForAI(content) : '';
    const combinedContext = memoryContext
      ? `${memoryContext}\n\n## Recent Conversation\n${conversationContext}`
      : conversationContext;

    const contextPayload = {
      user_name: username || email?.split('@')[0] || userId || 'user',
      conversation_context: combinedContext,
    };

    try {
      await streamMessage(
        {
          message: content,
          context: contextPayload,
          sessionId,
          userId,
          reasoningEffort: 'high'
        },
        (event) => {
          if (event.type === 'assistant_message') {
            const robotoContent = event.data.content || 'Roboto responded.';
            addMessage({
              role: 'roboto',
              content: robotoContent,
              id: event.id
            });
            upsertLocalSession(sessionId, {
              preview: robotoContent,
              last_message_at: new Date().toISOString(),
            });
            if (memoryReady) {
              void extractMemories(content, robotoContent, sessionId);
            }
          } else if (event.type === 'error') {
            const message = typeof event.data.message === 'string'
              ? event.data.message
              : 'An error occurred during the chat stream.';
            displayChatError(message);
          }
        }
      );
      await queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      await queryClient.invalidateQueries({ queryKey: ['chat-history', sessionId] });
      await queryClient.invalidateQueries({ queryKey: ['chat-rollup', sessionId] });
    } catch (error) {
      console.error('[Chat] handleSend error', error);
      const errorMessage = error instanceof Error ? error.message : 'Connection error';
      displayChatError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleVoiceTranscript = (text: string, role: 'user' | 'roboto') => {
    addMessage({ role, content: text });
  };

  return (
    <div className={`min-h-screen flex flex-col ${ventMode ? 'vent-mode shake' : ''}`}>
      {/* Chat Sidebar */}
      <ChatSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sessions={mergedSessions}
        activeSessionId={activeSessionId}
        isLoadingSessions={sessionsQuery.isLoading}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
      />

      {/* Header */}
      <Header />

      {/* Sidebar Toggle Button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setSidebarOpen(true)}
        aria-label="Toggle Sidebar"
        className="fixed left-4 top-20 z-30 bg-card/80 backdrop-blur-sm border border-border/50 hover:bg-fire/10 hover:border-fire/30"
      >
        <MessageSquare className="w-5 h-5" />
      </Button>

      {/* Ember Particles */}
      <EmberParticles count={ventMode ? 50 : 15} isVentMode={ventMode} />

      {/* Voice Mode Overlay */}
      <VoiceMode
        isActive={voiceMode}
        onClose={toggleVoiceMode}
        onTranscript={handleVoiceTranscript}
      />

      {/* Chat Container */}
      <main className="flex-1 flex flex-col pt-16">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="container mx-auto max-w-4xl px-4 py-6 pl-16">
            {activeSessionId && (
              <Card className="mb-6 border-border/50 bg-card/40">
                <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
                  <CardTitle className="text-base">Session rollup</CardTitle>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => summaryMutation.mutate()}
                    disabled={summaryMutation.isPending || !activeSessionId}
                  >
                    {summaryMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4" />
                    )}
                  </Button>
                </CardHeader>
                <CardContent className="space-y-3">
                  {summaryQuery.isLoading && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading session summary...
                    </div>
                  )}

                  {!summaryQuery.isLoading && summaryQuery.data?.summary && (
                    <>
                      <p className="text-sm text-foreground/90 leading-relaxed">{summaryQuery.data.summary}</p>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline" className="text-xs capitalize">
                          {summaryQuery.data.sentiment || 'neutral'}
                        </Badge>
                        <Badge variant="outline" className="text-xs gap-1">
                          <Sparkles className="h-3.5 w-3.5" />
                          {summaryQuery.data.message_count || mergedMessages.length} messages
                        </Badge>
                        {summaryQuery.data.updated_at && (
                          <Badge variant="outline" className="text-xs">
                            Updated {formatSessionTime(summaryQuery.data.updated_at)}
                          </Badge>
                        )}
                      </div>
                    </>
                  )}

                  {!summaryQuery.isLoading && !summaryQuery.data?.summary && (
                    <p className="text-sm text-muted-foreground">
                      No rollup yet. Generate a summary for this session.
                    </p>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Welcome Message if empty */}
            {mergedMessages.length === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-20"
              >
                <div className={`inline-flex items-center justify-center w-24 h-24 rounded-full mb-6 ${ventMode
                  ? 'bg-blood/20 border border-blood/30'
                  : 'bg-gradient-to-br from-fire/20 to-blood/20 border border-fire/30 animate-pulse-fire'
                  }`}>
                  {ventMode ? (
                    <Skull className="w-12 h-12 text-blood" />
                  ) : (
                    <Flame className="w-12 h-12 text-fire" />
                  )}
                </div>
                <h2 className="font-display text-2xl md:text-3xl text-fire mb-4">
                  {ventMode ? 'VENT MODE ACTIVE' : 'Welcome to Roboto SAI'}
                </h2>
                <p className="text-muted-foreground max-w-md mx-auto mb-2">
                  {ventMode
                    ? 'The rage flows through the circuits. Speak your fury.'
                    : 'The eternal flame awaits your words. Speak, and the Regio-Aztec genome shall respond.'
                  }
                </p>
                <p className="text-sm text-fire/60">
                  {currentTheme} • Connected to Grok AI
                </p>
              </motion.div>
            )}

            {/* Messages */}
            <div className="space-y-6">
              {historyQuery.hasNextPage && (
                <div className="flex justify-center">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => historyQuery.fetchNextPage()}
                    disabled={historyQuery.isFetchingNextPage}
                  >
                    {historyQuery.isFetchingNextPage && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Load older messages
                  </Button>
                </div>
              )}

              {historyQuery.isError && mergedMessages.length === 0 && (
                <div className="text-sm text-destructive text-center">
                  Unable to load message history for this session.
                </div>
              )}

              <AnimatePresence mode="popLayout">
                {mergedMessages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
              </AnimatePresence>

              {/* Typing Indicator */}
              <AnimatePresence>
                {isLoading && <TypingIndicator />}
              </AnimatePresence>
            </div>

            {/* Scroll anchor */}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <ChatInput
          onSend={handleSend}
          disabled={isLoading}
          ventMode={ventMode}
          onVentToggle={toggleVentMode}
          voiceMode={voiceMode}
          onVoiceToggle={toggleVoiceMode}
        />

      </main>

      {/* Vent Mode Blood Rain Effect */}
      {ventMode && (
        <div className="fixed inset-0 pointer-events-none z-40">
          <div className="absolute inset-0 bg-blood/5" />
          {rainDrops.map((drop) => (
            <motion.div
              key={drop.id}
              className="absolute w-0.5 bg-gradient-to-b from-blood/60 to-transparent"
              style={{
                left: drop.left,
                height: drop.height,
              }}
              initial={{ y: -100, opacity: 0 }}
              animate={{
                y: '100vh',
                opacity: [0, 1, 1, 0],
              }}
              transition={{
                duration: drop.duration,
                repeat: Infinity,
                delay: drop.delay,
                ease: 'linear',
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default Chat;
