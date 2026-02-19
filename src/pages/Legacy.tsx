/**
 * Roboto SAI Legacy Console
 * Production-grade chat interface with session management, summaries, and infinite history.
 */

import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { format, formatDistanceToNow, isSameDay } from 'date-fns';
import {
  AlertTriangle,
  Bot,
  Loader2,
  MessageSquare,
  PanelLeft,
  PanelRight,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Tag,
  Smile,
  Meh,
  Frown,
  Flame,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Header } from '@/components/layout/Header';
import { EmberParticles } from '@/components/effects/EmberParticles';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { TypingIndicator } from '@/components/chat/TypingIndicator';
import { useChatStore, Message } from '@/stores/chatStore';
import { useMemoryStore } from '@/stores/memoryStore';
import { useAuthStore } from '@/stores/authStore';
import { useToast } from '@/components/ui/use-toast';
import { useRobotoClient } from '@/hooks/useRobotoClient';
import { cn } from '@/lib/utils';

type SessionListItem = {
  id: string;
  title: string;
  preview: string;
  summary_preview?: string;
  last_message_at?: string;
  message_count?: number;
  topics?: string[];
  sentiment?: string;
  localOnly?: boolean;
};

type SummaryData = {
  summary: string;
  topics: string[];
  sentiment: string;
  message_count: number;
  updated_at?: string;
};

type HistoryPage = {
  messages: Message[];
  nextCursor?: string | null;
};

type RenderItem =
  | { type: 'divider'; id: string; label: string }
  | { type: 'message'; id: string; message: Message };

const HISTORY_PAGE_SIZE = 40;
const WINDOW_SIZE = 160;
const WINDOW_STEP = 60;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const toStringValue = (value: unknown): string =>
  typeof value === 'string' ? value : '';

const parseTimestamp = (value: unknown): Date => {
  if (value instanceof Date) return value;
  if (typeof value === 'string' || typeof value === 'number') {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }
  return new Date();
};

const normalizeRole = (role: unknown): Message['role'] => {
  if (role === 'assistant' || role === 'roboto') return 'roboto';
  return 'user';
};

const normalizeMessage = (value: unknown): Message | null => {
  if (!isRecord(value)) return null;
  const id = toStringValue(value.id) || toStringValue(value.message_id) || crypto.randomUUID();
  const timestamp = parseTimestamp(value.timestamp ?? value.created_at ?? value.createdAt ?? value.time);
  return {
    id,
    role: normalizeRole(value.role),
    content: toStringValue(value.content),
    timestamp,
    session_id: toStringValue(value.session_id) || toStringValue(value.sessionId) || undefined,
    user_id: toStringValue(value.user_id) || undefined,
    emotion: toStringValue(value.emotion) || undefined,
    emotion_text: toStringValue(value.emotion_text) || undefined,
    emotion_probabilities: isRecord(value.emotion_probabilities)
      ? (value.emotion_probabilities as Record<string, number>)
      : undefined,
  };
};

const normalizeSessions = (data: unknown): SessionListItem[] => {
  const list = Array.isArray(data)
    ? data
    : isRecord(data)
      ? (Array.isArray(data.sessions) ? data.sessions
        : Array.isArray(data.items) ? data.items
          : Array.isArray(data.data) ? data.data
            : [])
      : [];

  return list
    .map((item) => {
      if (!isRecord(item)) return null;
      const id = toStringValue(item.id)
        || toStringValue(item.session_id)
        || toStringValue(item.sessionId)
        || crypto.randomUUID();
      const title = toStringValue(item.title)
        || toStringValue(item.name)
        || toStringValue(item.session_title)
        || 'New chat';
      const preview = toStringValue(item.preview)
        || toStringValue(item.last_message)
        || toStringValue(item.last_message_preview)
        || toStringValue(item.summary_preview)
        || 'No messages yet';
      const summaryPreview = toStringValue(item.summary_preview)
        || toStringValue(item.summaryPreview)
        || '';
      const lastMessageAt = toStringValue(item.last_message_at)
        || toStringValue(item.last_message_time)
        || toStringValue(item.updated_at)
        || toStringValue(item.lastMessageAt);
      const messageCount = typeof item.message_count === 'number'
        ? item.message_count
        : typeof item.messageCount === 'number'
          ? item.messageCount
          : undefined;
      const topics = Array.isArray(item.topics)
        ? item.topics.filter((topic): topic is string => typeof topic === 'string')
        : Array.isArray(item.key_topics)
          ? item.key_topics.filter((topic): topic is string => typeof topic === 'string')
          : undefined;
      const sentiment = toStringValue(item.sentiment) || undefined;
      return {
        id,
        title,
        preview,
        summary_preview: summaryPreview || undefined,
        last_message_at: lastMessageAt || undefined,
        message_count: messageCount,
        topics,
        sentiment,
      };
    })
    .filter(Boolean) as SessionListItem[];
};

const normalizeHistory = (data: unknown): HistoryPage => {
  if (Array.isArray(data)) {
    return {
      messages: data.map(normalizeMessage).filter((msg): msg is Message => msg !== null),
    };
  }

  if (!isRecord(data)) return { messages: [] };

  const list = Array.isArray(data.messages)
    ? data.messages
    : Array.isArray(data.items)
      ? data.items
      : Array.isArray(data.data)
        ? data.data
        : [];
  const messages = list.map(normalizeMessage).filter((msg): msg is Message => msg !== null);
  const nextCursor = toStringValue(data.next_cursor)
    || toStringValue(data.nextCursor)
    || toStringValue(data.cursor)
    || (isRecord(data.pagination) ? toStringValue(data.pagination.next_cursor) : '');
  return {
    messages,
    nextCursor: nextCursor || null,
  };
};

const normalizeSummary = (data: unknown): SummaryData => {
  const payload = isRecord(data) && isRecord(data.rollup)
    ? data.rollup
    : isRecord(data) && isRecord(data.summary)
      ? data.summary
      : data;
  const summary = isRecord(payload) ? toStringValue(payload.summary)
    || toStringValue(payload.rollup)
    || toStringValue(payload.text)
    : '';
  const topics = isRecord(payload)
    ? Array.isArray(payload.topics)
      ? payload.topics.filter((topic): topic is string => typeof topic === 'string')
      : Array.isArray(payload.key_topics)
        ? payload.key_topics.filter((topic): topic is string => typeof topic === 'string')
        : []
    : [];
  const sentiment = isRecord(payload)
    ? toStringValue(payload.sentiment) || 'neutral'
    : 'neutral';
  const messageCount = isRecord(payload)
    ? typeof payload.message_count === 'number'
      ? payload.message_count
      : typeof payload.messageCount === 'number'
        ? payload.messageCount
        : 0
    : 0;
  const updatedAt = isRecord(payload)
    ? toStringValue(payload.updated_at) || toStringValue(payload.updatedAt)
    : '';

  return {
    summary,
    topics,
    sentiment,
    message_count: messageCount,
    updated_at: updatedAt || undefined,
  };
};

const formatSessionTime = (value?: string): string => {
  if (!value) return 'New';
  const parsed = parseTimestamp(value);
  if (Number.isNaN(parsed.getTime())) return 'Unknown';
  return formatDistanceToNow(parsed, { addSuffix: true });
};

const formatDayLabel = (value: Date): string => format(value, 'EEE, MMM d');

const buildTitle = (content: string): string => {
  const cleaned = content.replace(/[^\w\s]/g, '').trim();
  const words = cleaned.split(/\s+/).slice(0, 5);
  return words.join(' ') || 'New chat';
};

const useDebouncedValue = <T,>(value: T, delay = 300): T => {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(handle);
  }, [value, delay]);

  return debounced;
};

const useKeyboardShortcut = (handler: () => void) => {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'n') {
        event.preventDefault();
        handler();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handler]);
};

class LegacyErrorBoundary extends React.Component<
  { title?: string; onRetry?: () => void; children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  state = { hasError: false, error: undefined } as { hasError: boolean; error?: Error };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined });
    this.props.onRetry?.();
  };

  render() {
    if (this.state.hasError) {
      return (
        <PanelErrorState
          title={this.props.title || 'Panel error'}
          description={this.state.error?.message || 'Something went wrong.'}
          onRetry={this.handleRetry}
        />
      );
    }

    return this.props.children;
  }
}

const PanelErrorState = ({
  title,
  description,
  onRetry,
}: {
  title: string;
  description: string;
  onRetry?: () => void;
}) => (
  <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-border/50 bg-card/40 p-6 text-center">
    <AlertTriangle className="mb-3 h-6 w-6 text-destructive" />
    <p className="text-sm font-semibold text-foreground">{title}</p>
    <p className="text-xs text-muted-foreground max-w-xs">{description}</p>
    {onRetry && (
      <Button onClick={onRetry} variant="outline" size="sm" className="mt-4">
        Retry
      </Button>
    )}
  </div>
);

const PanelEmptyState = ({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
}) => (
  <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-border/50 bg-card/30 p-6 text-center">
    <MessageSquare className="mb-3 h-6 w-6 text-muted-foreground" />
    <p className="text-sm font-semibold text-foreground">{title}</p>
    <p className="text-xs text-muted-foreground max-w-xs">{description}</p>
    {action}
  </div>
);

const SessionRow = memo(({
  session,
  isActive,
  onSelect,
}: {
  session: SessionListItem;
  isActive: boolean;
  onSelect: (id: string) => void;
}) => {
  return (
    <button
      type="button"
      onClick={() => onSelect(session.id)}
      className={cn(
        'w-full rounded-xl border px-3 py-3 text-left transition-all',
        isActive
          ? 'border-primary/40 bg-primary/10 shadow-sm'
          : 'border-border/50 bg-card/40 hover:border-primary/30 hover:bg-card/60'
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold text-foreground truncate">{session.title}</p>
        <span className="text-[10px] text-muted-foreground whitespace-nowrap">
          {formatSessionTime(session.last_message_at)}
        </span>
      </div>
      <p className="mt-1 text-xs text-muted-foreground truncate">
        {session.preview}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-muted-foreground">
        {typeof session.message_count === 'number' && (
          <span>{session.message_count} msgs</span>
        )}
        {session.sentiment && (
          <span className="capitalize">{session.sentiment}</span>
        )}
        {session.localOnly && (
          <span className="text-fire">Draft</span>
        )}
      </div>
    </button>
  );
});

SessionRow.displayName = 'SessionRow';

const SessionsPanel = ({
  sessions,
  activeSessionId,
  searchValue,
  onSearchChange,
  onSelectSession,
  onNewSession,
  isLoading,
  isError,
  onRetry,
}: {
  sessions: SessionListItem[];
  activeSessionId: string | null;
  searchValue: string;
  onSearchChange: (value: string) => void;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  isLoading: boolean;
  isError: boolean;
  onRetry?: () => void;
}) => (
  <div className="flex h-full flex-col gap-4">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Sessions</p>
        <p className="text-sm font-semibold text-foreground">Your timeline</p>
      </div>
      <Button size="sm" className="btn-ember" onClick={onNewSession} aria-label="Create new session">
        <Plus className="h-4 w-4" />
      </Button>
    </div>
    <div className="relative">
      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        value={searchValue}
        onChange={(event) => onSearchChange(event.target.value)}
        placeholder="Search sessions"
        aria-label="Search sessions"
        className="pl-9"
      />
    </div>
    <Separator />
    <div aria-label="Session list" className="flex-1 space-y-3 overflow-y-auto pr-1">
      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, index) => (
            <Skeleton key={index} className="h-20 w-full" />
          ))}
        </div>
      )}
      {!isLoading && isError && (
        <PanelErrorState
          title="Unable to load sessions"
          description="The session list could not be retrieved."
          onRetry={onRetry}
        />
      )}
      {!isLoading && !isError && sessions.length === 0 && (
        <PanelEmptyState
          title="No sessions yet"
          description="Start a new conversation to build your history."
          action={
            <Button size="sm" className="mt-4" onClick={onNewSession}>
              Create session
            </Button>
          }
        />
      )}
      {!isLoading && !isError && sessions.map((session) => (
        <SessionRow
          key={session.id}
          session={session}
          isActive={session.id === activeSessionId}
          onSelect={onSelectSession}
        />
      ))}
    </div>
  </div>
);

const SentimentBadge = ({ sentiment }: { sentiment?: string }) => {
  const normalized = (sentiment || 'neutral').toLowerCase();
  const config =
    normalized === 'positive'
      ? { label: 'Positive', className: 'bg-emerald-500/10 text-emerald-300 border-emerald-400/30', Icon: Smile }
      : normalized === 'negative'
        ? { label: 'Negative', className: 'bg-rose-500/10 text-rose-300 border-rose-400/30', Icon: Frown }
        : { label: 'Neutral', className: 'bg-slate-500/10 text-slate-300 border-slate-400/30', Icon: Meh };
  return (
    <Badge variant="outline" className={cn('gap-1 border text-xs', config.className)}>
      <config.Icon className="h-3.5 w-3.5" />
      {config.label}
    </Badge>
  );
};

const SummaryPanel = ({
  summary,
  isLoading,
  isError,
  onRetry,
  onRegenerate,
  isRegenerating,
  messageCount,
}: {
  summary?: SummaryData;
  isLoading: boolean;
  isError: boolean;
  onRetry?: () => void;
  onRegenerate: () => void;
  isRegenerating: boolean;
  messageCount: number;
}) => (
  <div className="flex h-full flex-col gap-4">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Summary</p>
        <p className="text-sm font-semibold text-foreground">Session intelligence</p>
      </div>
      <button
        type="button"
        onClick={onRegenerate}
        disabled={isRegenerating}
        aria-label="Regenerate summary"
      >
        {isRegenerating ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <RefreshCw className="h-4 w-4" />
        )}
      </button>
    </div>
    {isLoading && (
      <div className="space-y-4">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-6 w-2/3" />
        <Skeleton className="h-6 w-1/2" />
      </div>
    )}
    {!isLoading && isError && (
      <PanelErrorState
        title="Summary unavailable"
        description="We could not load the rollup for this session."
        onRetry={onRetry}
      />
    )}
    {!isLoading && !isError && summary && summary.summary && (
      <Card className="border-border/50 bg-card/40">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Current rollup</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-foreground/90 leading-relaxed">
            {summary.summary}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <SentimentBadge sentiment={summary.sentiment} />
            <Badge variant="outline" className="gap-1 border-border/60 text-xs">
              <Sparkles className="h-3.5 w-3.5" />
              {messageCount} messages
            </Badge>
            {summary.updated_at && (
              <Badge variant="outline" className="border-border/60 text-xs">
                Updated {formatSessionTime(summary.updated_at)}
              </Badge>
            )}
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground mb-2 flex items-center gap-1">
              <Tag className="h-3 w-3" />
              Key topics
            </p>
            <div className="flex flex-wrap gap-2">
              {summary.topics.length > 0
                ? summary.topics.map((topic) => (
                  <Badge key={topic} variant="secondary" className="text-xs">
                    {topic}
                  </Badge>
                ))
                : (
                  <Badge variant="outline" className="text-xs">
                    No topics yet
                  </Badge>
                )}
            </div>
          </div>
        </CardContent>
      </Card>
    )}
    {!isLoading && !isError && (!summary || !summary.summary) && (
      <PanelEmptyState
        title="No summary yet"
        description="Generate a rollup to unlock key themes and sentiment."
        action={
          <Button size="sm" className="mt-4" onClick={onRegenerate}>
            Generate summary
          </Button>
        }
      />
    )}
  </div>
);

const Legacy = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();
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
    createNewConversation,
    selectConversation,
    currentConversationId,
  } = useChatStore();
  const {
    buildContextForAI,
    addMemory,
    addConversationSummary,
    trackEntity,
    isReady: memoryReady,
  } = useMemoryStore();
  const { sendMessage: streamMessage, isConnected } = useRobotoClient();

  const [searchValue, setSearchValue] = useState('');
  const debouncedSearch = useDebouncedValue(searchValue, 350);
  const [localSessions, setLocalSessions] = useState<SessionListItem[]>([]);
  const [sessionOverrides, setSessionOverrides] = useState<Record<string, Partial<SessionListItem>>>({});
  const [isAutoScroll, setIsAutoScroll] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);
  const [windowStart, setWindowStart] = useState(0);
  const previousScrollHeightRef = useRef<number | null>(null);
  const previousScrollTopRef = useRef<number | null>(null);

  const apiBaseUrl = useMemo(() => {
    const envUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '';
    const fallbackBase = globalThis.window?.location.origin ?? '';
    return (envUrl || fallbackBase).replace(/\/+$/, '').replace(/\/api$/, '');
  }, []);

  const apiFetch = useCallback(async (path: string, init?: RequestInit) => {
    const response = await fetch(`${apiBaseUrl}${path}`, {
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers || {}),
      },
      ...init,
    });

    if (response.status === 401) {
      toast({
        variant: 'destructive',
        title: 'Authentication required',
        description: 'Please log in to continue.',
      });
      navigate('/login');
      throw new Error('Unauthorized');
    }

    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        const data = await response.json();
        if (isRecord(data)) {
          message = toStringValue(data.detail) || toStringValue(data.error) || message;
        }
      } catch {
        // ignore JSON parsing errors
      }
      throw new Error(message);
    }

    return response.json();
  }, [apiBaseUrl, navigate, toast]);

  const sessionsQuery = useQuery({
    queryKey: ['sessions', debouncedSearch],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (debouncedSearch) params.set('search', debouncedSearch);
      const path = params.toString() ? `/api/sessions?${params}` : '/api/sessions';
      const data = await apiFetch(path);
      return normalizeSessions(data);
    },
    staleTime: 10000,
  });

  const mergedSessions = useMemo(() => {
    const base = sessionsQuery.data ?? [];
    const sessionMap = new Map<string, SessionListItem>(base.map((item) => [item.id, item]));
    localSessions.forEach((item) => {
      if (!sessionMap.has(item.id)) {
        sessionMap.set(item.id, item);
      }
    });
    const merged = Array.from(sessionMap.values()).map((item) => ({
      ...item,
      ...(sessionOverrides[item.id] || {}),
    }));
    return merged.sort((a, b) => {
      const timeA = a.last_message_at ? parseTimestamp(a.last_message_at).getTime() : 0;
      const timeB = b.last_message_at ? parseTimestamp(b.last_message_at).getTime() : 0;
      return timeB - timeA;
    });
  }, [sessionsQuery.data, localSessions, sessionOverrides]);

  const activeSessionId = currentConversationId || mergedSessions[0]?.id || null;

  useEffect(() => {
    if (!currentConversationId && mergedSessions[0]?.id) {
      selectConversation(mergedSessions[0].id);
    }
  }, [currentConversationId, mergedSessions, selectConversation]);

  useEffect(() => {
    if (!sessionsQuery.data) return;
    setLocalSessions((prev) => prev.filter((item) => !sessionsQuery.data.some((session) => session.id === item.id)));
  }, [sessionsQuery.data]);

  const historyQuery = useInfiniteQuery<
    HistoryPage,
    Error,
    HistoryPage,
    [string, string | null],
    string | undefined
  >({
    queryKey: ['chat-history', activeSessionId],
    enabled: Boolean(activeSessionId),
    initialPageParam: undefined,
    queryFn: async ({ pageParam }) => {
      const params = new URLSearchParams();
      params.set('limit', String(HISTORY_PAGE_SIZE));
      if (pageParam) params.set('cursor', String(pageParam));
      if (activeSessionId) params.set('session_id', activeSessionId);
      try {
        const data = await apiFetch(`/api/chat/history?${params}`);
        return normalizeHistory(data);
      } catch (error) {
        if (error instanceof Error && /404/.test(error.message)) {
          const fallbackData = await apiFetch(`/api/chat/history/paginated?${params}`);
          return normalizeHistory(fallbackData);
        }
        throw error;
      }
    },
    getNextPageParam: (lastPage) => lastPage.nextCursor || undefined,
    retry: 2,
  });

  const historyMessages = useMemo(() => {
    const pages = historyQuery.data?.pages ?? [];
    const combined = pages.flatMap((page) => page.messages || []);
    return combined.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }, [historyQuery.data]);

  const storeMessages = useMemo(() => getMessages(), [getMessages]);

  const mergedMessages = useMemo(() => {
    const map = new Map<string, Message>();
    [...historyMessages, ...storeMessages].forEach((message) => {
      map.set(message.id, {
        ...message,
        timestamp: parseTimestamp(message.timestamp),
      });
    });
    return Array.from(map.values()).sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }, [historyMessages, storeMessages]);

  const renderItems = useMemo<RenderItem[]>(() => {
    const items: RenderItem[] = [];
    let lastDay: Date | null = null;
    mergedMessages.forEach((message) => {
      if (!lastDay || !isSameDay(message.timestamp, lastDay)) {
        const label = formatDayLabel(message.timestamp);
        items.push({ type: 'divider', id: `divider-${message.timestamp.toISOString()}`, label });
        lastDay = message.timestamp;
      }
      items.push({ type: 'message', id: message.id, message });
    });
    return items;
  }, [mergedMessages]);

  const visibleItems = renderItems.slice(windowStart);

  useEffect(() => {
    if (!isAutoScroll) return;
    setWindowStart(Math.max(0, renderItems.length - WINDOW_SIZE));
  }, [renderItems.length, isAutoScroll]);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    if (previousScrollHeightRef.current !== null && previousScrollTopRef.current !== null) {
      const heightDiff = node.scrollHeight - previousScrollHeightRef.current;
      node.scrollTop = previousScrollTopRef.current + heightDiff;
      previousScrollHeightRef.current = null;
      previousScrollTopRef.current = null;
    }
  }, [historyQuery.data?.pages.length, windowStart]);

  const scrollToBottom = useCallback(() => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, []);

  useEffect(() => {
    setIsAutoScroll(true);
    requestAnimationFrame(scrollToBottom);
  }, [activeSessionId, scrollToBottom]);

  useEffect(() => {
    if (isAutoScroll) {
      requestAnimationFrame(scrollToBottom);
    }
  }, [mergedMessages.length, isLoading, isAutoScroll, scrollToBottom]);

  const summaryQuery = useQuery({
    queryKey: ['summary', activeSessionId],
    enabled: Boolean(activeSessionId),
    queryFn: async () => {
      const params = activeSessionId ? `?session_id=${activeSessionId}` : '';
      const data = await apiFetch(`/api/conversations/rollup${params}`);
      return normalizeSummary(data);
    },
    retry: 2,
  });

  const summaryMutation = useMutation({
    mutationFn: async () => {
      if (!activeSessionId) return null;
      return apiFetch(`/api/conversations/summarize?session_id=${encodeURIComponent(activeSessionId)}`, {
        method: 'POST',
      });
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['summary', activeSessionId] });
      toast({
        title: 'Summary refreshed',
        description: 'Conversation rollup updated.',
      });
    },
    onError: (error) => {
      toast({
        variant: 'destructive',
        title: 'Summary failed',
        description: error instanceof Error ? error.message : 'Unable to regenerate summary.',
      });
    },
  });

  const updateSessionOverride = useCallback((sessionId: string, patch: Partial<SessionListItem>) => {
    setSessionOverrides((prev) => ({
      ...prev,
      [sessionId]: {
        ...(prev[sessionId] || {}),
        ...patch,
      },
    }));
  }, []);

  const handleNewSession = useCallback(() => {
    const newId = createNewConversation();
    const now = new Date().toISOString();
    setLocalSessions((prev) => [
      {
        id: newId,
        title: 'New chat',
        preview: 'No messages yet',
        last_message_at: now,
        message_count: 0,
        localOnly: true,
      },
      ...prev,
    ]);
    selectConversation(newId);
    toast({
      title: 'New session created',
      description: 'Ready for the next conversation.',
    });
  }, [createNewConversation, selectConversation, toast]);

  useKeyboardShortcut(handleNewSession);

  const extractMemories = useCallback(async (userMessage: string, robotoResponse: string, sessionId: string) => {
    const namePattern = /(?:my (?:name is|friend|brother|sister|mom|dad|wife|husband|partner|boss|colleague) (?:is )?|I'm |I am )([A-Z][a-z]+)/gi;
    let match;
    while ((match = namePattern.exec(userMessage)) !== null) {
      const entityName = match[1];
      const entityType = userMessage.toLowerCase().includes('name is') ? 'self' : 'person';
      await trackEntity(entityName, entityType, userMessage);
    }

    const preferencePatterns = [
      { pattern: /I (?:really )?(?:love|like|prefer|enjoy) (.+?)(?:\.|,|!|$)/i, type: 'likes' },
      { pattern: /I (?:hate|dislike|don't like|can't stand) (.+?)(?:\.|,|!|$)/i, type: 'dislikes' },
      { pattern: /I'm (?:a|an) (.+?)(?:\.|,|!|$)/i, type: 'identity' },
    ];

    for (const { pattern, type } of preferencePatterns) {
      const prefMatch = userMessage.match(pattern);
      if (prefMatch?.[1]) {
        await addMemory(
          `User ${type}: ${prefMatch[1]}`,
          'preferences',
          1.2,
          { source: sessionId, extractedFrom: userMessage }
        );
      }
    }

    if (robotoResponse.trim().length > 0) {
      await addConversationSummary(sessionId, robotoResponse.slice(0, 220));
    }
  }, [addConversationSummary, addMemory, trackEntity]);

  const displayChatError = useCallback((message: string) => {
    toast({
      variant: 'destructive',
      title: 'Connection error',
      description: message || 'The response stream was interrupted.',
    });
  }, [toast]);

  const handleSend = useCallback(async (content: string, attachments?: Message['attachments']) => {
    if (!isLoggedIn || !userId) {
      navigate('/login');
      return;
    }

    setLoading(true);
    const conversationId = addMessage({ role: 'user', content, attachments });
    updateSessionOverride(conversationId, {
      title: buildTitle(content),
      preview: content,
      last_message_at: new Date().toISOString(),
      message_count: (sessionOverrides[conversationId]?.message_count || 0) + 1,
    });

    const conversationContext = useChatStore.getState().getAllConversationsContext();
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
          sessionId: conversationId,
          userId,
          reasoningEffort: 'high',
        },
        (event) => {
          if (event.type === 'assistant_message') {
            const data = event.data as { content?: string } | undefined;
            const robotoContent = data?.content || 'Roboto responded.';
            addMessage({
              role: 'roboto',
              content: robotoContent,
              id: event.id,
            });
            updateSessionOverride(conversationId, {
              preview: robotoContent,
              last_message_at: new Date().toISOString(),
            });
            if (memoryReady) {
              void extractMemories(content, robotoContent, conversationId);
            }
          } else if (event.type === 'error') {
            const data = event.data as { message?: string } | undefined;
            const errorMessage = typeof data?.message === 'string'
              ? data.message
              : 'An error occurred during the chat stream.';
            displayChatError(errorMessage);
          }
        }
      );
      queryClient.invalidateQueries({ queryKey: ['summary', conversationId] });
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Connection error';
      displayChatError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [
    addMessage,
    buildContextForAI,
    displayChatError,
    email,
    extractMemories,
    isLoggedIn,
    memoryReady,
    navigate,
    queryClient,
    sessionOverrides,
    setLoading,
    streamMessage,
    updateSessionOverride,
    userId,
    username,
  ]);

  const handleSessionSelect = useCallback((id: string) => {
    selectConversation(id);
  }, [selectConversation]);

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    const node = event.currentTarget;
    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    setIsAutoScroll(distanceFromBottom < 200);

    if (node.scrollTop < 120) {
      if (windowStart > 0) {
        previousScrollHeightRef.current = node.scrollHeight;
        previousScrollTopRef.current = node.scrollTop;
        setWindowStart((prev) => Math.max(0, prev - WINDOW_STEP));
        return;
      }
      if (historyQuery.hasNextPage && !historyQuery.isFetchingNextPage) {
        previousScrollHeightRef.current = node.scrollHeight;
        previousScrollTopRef.current = node.scrollTop;
        historyQuery.fetchNextPage();
      }
    }
  }, [historyQuery, windowStart]);

  const headerUserLabel = username || email || 'User';
  const headerInitials = headerUserLabel
    .split(' ')
    .map((part) => part.charAt(0).toUpperCase())
    .slice(0, 2)
    .join('');

  return (
    <div className={cn('min-h-screen flex flex-col bg-background', ventMode ? 'vent-mode shake' : '')}>
      <Header />
      <EmberParticles count={ventMode ? 35 : 18} isVentMode={ventMode} />

      <main className="flex-1 pt-16">
        <div className="mx-auto w-full max-w-[1500px] px-4 py-6">
          <div className="flex flex-col gap-4 rounded-2xl border border-border/50 bg-card/30 p-4 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-fire/20 to-blood/20 border border-fire/30">
                <Flame className="h-6 w-6 text-fire" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                  Legacy console
                </p>
                <h1 className="text-xl font-semibold text-foreground">
                  Conversation Command
                </h1>
                <p className="text-xs text-muted-foreground">{currentTheme}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="hidden text-right md:block">
                <p className="text-sm font-medium text-foreground">{headerUserLabel}</p>
                <p className="text-xs text-muted-foreground">
                  {isConnected ? 'Connected' : 'Offline'}
                </p>
              </div>
              <Avatar>
                <AvatarFallback className="bg-muted text-sm font-semibold">
                  {headerInitials || 'RS'}
                </AvatarFallback>
              </Avatar>
            </div>
          </div>
        </div>

        <div className="mx-auto w-full max-w-[1500px] px-4 pb-8">
          <div className="flex items-center justify-between gap-3 pb-4 lg:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon" aria-label="Open sessions">
                  <PanelLeft className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-[320px] sm:w-[360px]">
                <SheetHeader>
                  <SheetTitle>Sessions</SheetTitle>
                </SheetHeader>
                <div className="mt-4 h-[calc(100vh-120px)]">
                  <LegacyErrorBoundary title="Sessions failed" onRetry={sessionsQuery.refetch}>
                    <SessionsPanel
                      sessions={mergedSessions}
                      activeSessionId={activeSessionId}
                      searchValue={searchValue}
                      onSearchChange={setSearchValue}
                      onSelectSession={handleSessionSelect}
                      onNewSession={handleNewSession}
                      isLoading={sessionsQuery.isLoading}
                      isError={sessionsQuery.isError}
                      onRetry={sessionsQuery.refetch}
                    />
                  </LegacyErrorBoundary>
                </div>
              </SheetContent>
            </Sheet>
            <Button variant="outline" size="icon" onClick={handleNewSession} aria-label="New session">
              <Plus className="h-5 w-5" />
            </Button>
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon" aria-label="Open summary">
                  <PanelRight className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[320px] sm:w-[360px]">
                <SheetHeader>
                  <SheetTitle>Summary</SheetTitle>
                </SheetHeader>
                <div className="mt-4 h-[calc(100vh-120px)]">
                  <LegacyErrorBoundary title="Summary failed" onRetry={summaryQuery.refetch}>
                    <SummaryPanel
                      summary={summaryQuery.data}
                      isLoading={summaryQuery.isLoading}
                      isError={summaryQuery.isError}
                      onRetry={summaryQuery.refetch}
                      onRegenerate={() => summaryMutation.mutate()}
                      isRegenerating={summaryMutation.isPending}
                      messageCount={mergedMessages.length}
                    />
                  </LegacyErrorBoundary>
                </div>
              </SheetContent>
            </Sheet>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[280px_minmax(0,1fr)] xl:grid-cols-[280px_minmax(0,1fr)_320px]">
            <aside className="hidden h-[calc(100vh-220px)] flex-col rounded-2xl border border-border/50 bg-card/30 p-4 lg:flex">
              <LegacyErrorBoundary title="Sessions failed" onRetry={sessionsQuery.refetch}>
                <SessionsPanel
                  sessions={mergedSessions}
                  activeSessionId={activeSessionId}
                  searchValue={searchValue}
                  onSearchChange={setSearchValue}
                  onSelectSession={handleSessionSelect}
                  onNewSession={handleNewSession}
                  isLoading={sessionsQuery.isLoading}
                  isError={sessionsQuery.isError}
                  onRetry={sessionsQuery.refetch}
                />
              </LegacyErrorBoundary>
            </aside>

            <section className="flex h-[calc(100vh-220px)] flex-col rounded-2xl border border-border/50 bg-card/30">
              <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
                <div className="flex items-center gap-3">
                  <Bot className="h-5 w-5 text-fire" />
                  <div>
                    <p className="text-sm font-semibold text-foreground">Live conversation</p>
                    <p className="text-xs text-muted-foreground">
                      {activeSessionId ? `Session ${activeSessionId.slice(0, 8)}` : 'Select a session'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {isConnected ? 'Online' : 'Offline'}
                  </Badge>
                  <Badge variant="outline" className="text-xs hidden md:inline-flex">
                    Ctrl + N
                  </Badge>
                </div>
              </div>

              <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto px-4 py-4"
                onScroll={handleScroll}
                aria-live="polite"
              >
                {historyQuery.isLoading && mergedMessages.length === 0 && (
                  <div className="space-y-4">
                    {Array.from({ length: 6 }).map((_, index) => (
                      <Skeleton key={index} className="h-20 w-full" />
                    ))}
                  </div>
                )}

                {!historyQuery.isLoading && historyQuery.isError && mergedMessages.length === 0 && (
                  <PanelErrorState
                    title="Unable to load messages"
                    description="The message history could not be retrieved."
                    onRetry={historyQuery.refetch}
                  />
                )}

                {!historyQuery.isLoading && !historyQuery.isError && mergedMessages.length === 0 && (
                  <PanelEmptyState
                    title="No messages yet"
                    description="Start a message to ignite this session."
                    action={
                      <Button size="sm" className="mt-4" onClick={handleNewSession}>
                        Start new session
                      </Button>
                    }
                  />
                )}

                {mergedMessages.length > 0 && (
                  <div className="space-y-4">
                    {historyQuery.isFetchingNextPage && (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading older messages
                      </div>
                    )}
                    <div className="space-y-4">
                      {visibleItems.map((item) =>
                        item.type === 'divider' ? (
                          <div key={item.id} className="flex items-center gap-3">
                            <Separator className="flex-1" />
                            <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                              {item.label}
                            </span>
                            <Separator className="flex-1" />
                          </div>
                        ) : (
                          <ChatMessage key={item.id} message={item.message} />
                        )
                      )}
                    </div>
                  </div>
                )}

                {isLoading && <TypingIndicator />}
              </div>

              <ChatInput
                onSend={handleSend}
                disabled={isLoading}
                ventMode={ventMode}
                onVentToggle={toggleVentMode}
                voiceMode={voiceMode}
                onVoiceToggle={toggleVoiceMode}
              />
            </section>

            <aside className="hidden h-[calc(100vh-220px)] flex-col rounded-2xl border border-border/50 bg-card/30 p-4 xl:flex">
              <LegacyErrorBoundary title="Summary failed" onRetry={summaryQuery.refetch}>
                <SummaryPanel
                  summary={summaryQuery.data}
                  isLoading={summaryQuery.isLoading}
                  isError={summaryQuery.isError}
                  onRetry={summaryQuery.refetch}
                  onRegenerate={() => summaryMutation.mutate()}
                  isRegenerating={summaryMutation.isPending}
                  messageCount={mergedMessages.length}
                />
              </LegacyErrorBoundary>
            </aside>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Legacy;
