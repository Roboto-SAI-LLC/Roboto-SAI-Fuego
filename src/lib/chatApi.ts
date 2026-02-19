import { Message } from '@/stores/chatStore';

export type ChatSessionListItem = {
  id: string;
  title: string;
  preview: string;
  summary_preview?: string;
  last_message_at?: string | number;
  message_count?: number;
  topics?: string[];
  sentiment?: string;
};

export type ChatHistoryPage = {
  messages: Message[];
  hasMore: boolean;
  nextCursor?: string | null;
};

export type ChatRollupSummary = {
  summary: string;
  topics: string[];
  sentiment: string;
  message_count: number;
  updated_at?: string | number;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const toStringValue = (value: unknown): string =>
  typeof value === 'string'
    ? value
    : typeof value === 'number' && Number.isFinite(value)
      ? String(value)
      : '';

const toTimestampValue = (value: unknown): string | number | undefined => {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.toISOString();
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim().length > 0) return value;
  return undefined;
};

const toNumberValue = (value: unknown): number | undefined =>
  typeof value === 'number'
    ? value
    : typeof value === 'string' && value.trim().length > 0
      ? Number.isFinite(Number(value))
        ? Number(value)
        : undefined
      : undefined;

const toBooleanValue = (value: unknown): boolean | undefined => {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (value === 1) return true;
    if (value === 0) return false;
    return undefined;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
    if (normalized === 'false' || normalized === '0' || normalized === 'no') return false;
  }
  return undefined;
};

const unwrapContainer = (value: unknown): unknown => {
  if (!isRecord(value)) return value;

  const dataValue = value.data;
  if (Array.isArray(dataValue) || isRecord(dataValue)) return dataValue;

  const resultValue = value.result;
  if (Array.isArray(resultValue) || isRecord(resultValue)) return resultValue;

  const payloadValue = value.payload;
  if (Array.isArray(payloadValue) || isRecord(payloadValue)) return payloadValue;

  return value;
};

const getPaginationRecord = (value: unknown): Record<string, unknown> | undefined => {
  if (!isRecord(value)) return undefined;
  if (isRecord(value.pagination)) return value.pagination;
  if (isRecord(value.page_info)) return value.page_info;
  if (isRecord(value.meta)) return value.meta;
  return undefined;
};

export const parseTimestamp = (value: unknown): Date => {
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

export const normalizeMessage = (value: unknown): Message | null => {
  if (!isRecord(value)) return null;

  const id = toStringValue(value.id)
    || toStringValue(value.message_id)
    || crypto.randomUUID();

  return {
    id,
    role: normalizeRole(value.role),
    content: toStringValue(value.content),
    timestamp: parseTimestamp(value.timestamp ?? value.created_at ?? value.createdAt ?? value.time),
    session_id: toStringValue(value.session_id) || toStringValue(value.sessionId) || undefined,
    user_id: toStringValue(value.user_id) || undefined,
    emotion: toStringValue(value.emotion) || undefined,
    emotion_text: toStringValue(value.emotion_text) || undefined,
    emotion_probabilities: isRecord(value.emotion_probabilities)
      ? (value.emotion_probabilities as Record<string, number>)
      : undefined,
  };
};

export const normalizeSessionsResponse = (data: unknown): ChatSessionListItem[] => {
  const payload = unwrapContainer(data);
  const list = Array.isArray(payload)
    ? payload
    : isRecord(payload)
      ? (Array.isArray(payload.sessions) ? payload.sessions
        : Array.isArray(payload.items) ? payload.items
          : Array.isArray(payload.data) ? payload.data
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

      const summaryPreview = toStringValue(item.summary_preview)
        || toStringValue(item.summaryPreview)
        || toStringValue(item.summary)
        || '';

      const preview = toStringValue(item.preview)
        || toStringValue(item.last_message)
        || toStringValue(item.last_message_preview)
        || summaryPreview
        || 'No messages yet';

      const lastMessageAt = toTimestampValue(item.last_message_at)
        || toTimestampValue(item.last_message_time)
        || toTimestampValue(item.last_message_ts)
        || toTimestampValue(item.last_message_timestamp)
        || toTimestampValue(item.lastMessageAt)
        || toTimestampValue(item.lastMessageTime)
        || toTimestampValue(item.lastMessageTimestamp)
        || toTimestampValue(item.updated_at)
        || toTimestampValue(item.updatedAt)
        || toTimestampValue(item.created_at)
        || toTimestampValue(item.createdAt);

      const messageCount = toNumberValue(item.message_count) ?? toNumberValue(item.messageCount);

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
    .filter(Boolean) as ChatSessionListItem[];
};

export const normalizeHistoryResponse = (data: unknown): ChatHistoryPage => {
  const payload = unwrapContainer(data);

  if (Array.isArray(payload)) {
    return {
      messages: payload.map(normalizeMessage).filter((msg): msg is Message => msg !== null),
      hasMore: false,
      nextCursor: null,
    };
  }

  if (!isRecord(payload)) {
    return {
      messages: [],
      hasMore: false,
      nextCursor: null,
    };
  }

  const list = Array.isArray(payload.messages)
    ? payload.messages
    : Array.isArray(payload.items)
      ? payload.items
      : Array.isArray(payload.history)
        ? payload.history
        : Array.isArray(payload.data)
          ? payload.data
          : [];

  const pagination = getPaginationRecord(payload);
  const paginationValues: Record<string, unknown> = pagination ?? {};

  let nextCursor = toStringValue(payload.next_cursor)
    || toStringValue(payload.nextCursor)
    || toStringValue(payload.cursor)
    || toStringValue(payload.next)
    || toStringValue(payload.next_page_token)
    || toStringValue(payload.nextPageToken)
    || toStringValue(paginationValues.next_cursor)
    || toStringValue(paginationValues.nextCursor)
    || toStringValue(paginationValues.cursor)
    || toStringValue(paginationValues.next)
    || toStringValue(paginationValues.next_page_token)
    || toStringValue(paginationValues.nextPageToken)
    || null;

  const nextPage = toNumberValue(payload.next_page)
    ?? toNumberValue(payload.nextPage)
    ?? toNumberValue(paginationValues.next_page)
    ?? toNumberValue(paginationValues.nextPage);

  if (!nextCursor && typeof nextPage === 'number') {
    nextCursor = String(nextPage);
  }

  const hasMoreFlag = toBooleanValue(payload.has_more)
    ?? toBooleanValue(payload.hasMore)
    ?? toBooleanValue(payload.has_next)
    ?? toBooleanValue(payload.hasNext)
    ?? toBooleanValue(payload.has_next_page)
    ?? toBooleanValue(payload.hasNextPage)
    ?? toBooleanValue(paginationValues.has_more)
    ?? toBooleanValue(paginationValues.hasMore)
    ?? toBooleanValue(paginationValues.has_next)
    ?? toBooleanValue(paginationValues.hasNext)
    ?? toBooleanValue(paginationValues.has_next_page)
    ?? toBooleanValue(paginationValues.hasNextPage);

  const hasMore = hasMoreFlag ?? (typeof nextPage === 'number' ? true : undefined) ?? Boolean(nextCursor);

  return {
    messages: list.map(normalizeMessage).filter((msg): msg is Message => msg !== null),
    hasMore,
    nextCursor,
  };
};

export const normalizeRollupResponse = (data: unknown): ChatRollupSummary => {
  const container = unwrapContainer(data);
  const payload = isRecord(container) && isRecord(container.rollup)
    ? container.rollup
    : isRecord(container) && isRecord(container.summary)
      ? container.summary
      : container;

  const summary = isRecord(payload)
    ? toStringValue(payload.summary) || toStringValue(payload.rollup) || toStringValue(payload.text)
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
    ? toNumberValue(payload.message_count) ?? toNumberValue(payload.messageCount) ?? 0
    : 0;

  const updatedAt = isRecord(payload)
    ? toTimestampValue(payload.updated_at) || toTimestampValue(payload.updatedAt)
    : undefined;

  return {
    summary,
    topics,
    sentiment,
    message_count: messageCount,
    updated_at: updatedAt,
  };
};
