/**
 * RobotoClient SDK Shared Types
 */

export type ChatEventType =
  | 'assistant_message'
  | 'error';

export interface AssistantMessageData {
  content: string;
  metadata?: Record<string, unknown>;
}

export interface ErrorEventData {
  message: string;
  code?: string;
  details?: unknown;
}

export interface ChatEvent<T = unknown> {
  type: ChatEventType;
  id: string;
  timestamp: number;
  data: T;
}

export type ChatEventPayload =
  | ChatEvent<AssistantMessageData>
  | ChatEvent<ErrorEventData>;

export interface RobotoClientConfig {
  backendBaseUrl: string;
  defaultHeaders?: Record<string, string>;
  userId?: string;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  context?: Record<string, unknown>;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}
