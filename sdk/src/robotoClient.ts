/**
 * RobotoClient SDK Implementation
 * Handles chat events and streaming responses.
 */

import {
  ChatEventPayload,
  RobotoClientConfig,
} from './types';

interface StreamChatParams {
  message: string;
  sessionId?: string;
  context?: Record<string, unknown>;
  reasoningEffort?: string;
  userId?: string;
}

export class RobotoClient {
  private readonly config: RobotoClientConfig;
  private readonly headers: Record<string, string>;

  constructor(config: RobotoClientConfig) {
    this.config = config;
    this.headers = {
      'Content-Type': 'application/json',
      ...(config.defaultHeaders ?? {})
    };
  }

  private generateId(): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    return `evt_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }

  private createErrorEvent(message: string, details?: unknown): ChatEventPayload {
    return {
      type: 'error',
      id: this.generateId(),
      timestamp: Date.now(),
      data: {
        message,
        details,
      }
    };
  }

  private normalizeEvent(raw: any): ChatEventPayload {
    const base = {
      id: typeof raw.id === 'string' ? raw.id : this.generateId(),
      timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
    };

    if (raw?.type === 'assistant_message') {
      return {
        ...base,
        type: 'assistant_message',
        data: {
          content: typeof raw.data?.content === 'string' ? raw.data.content : '',
          metadata: raw.data?.metadata ?? {}
        }
      };
    }

    return {
      ...base,
      type: 'error',
      data: {
        message: raw?.data?.message ?? 'Unknown event received',
        details: raw
      }
    };
  }

  async *streamChat(params: StreamChatParams): AsyncGenerator<ChatEventPayload> {
    const body = {
      message: params.message,
      session_id: params.sessionId,
      reasoning_effort: params.reasoningEffort ?? 'medium',
      context: params.context ?? {},
      user_id: params.userId ?? this.config.userId,
      timestamp: Date.now()
    };

    let payload: any;
    try {
      const response = await fetch(`${this.config.backendBaseUrl.replace(/\/$/, '')}/api/chat`, {
        method: 'POST',
        headers: this.headers,
        credentials: 'include',
        body: JSON.stringify(body)
      });

      payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.detail ?? payload?.error ?? 'Chat stream failed');
      }
    } catch (error) {
      yield this.createErrorEvent((error as Error).message, error);
      return;
    }

    const rawEvents = Array.isArray(payload?.events)
      ? payload.events
      : [
          {
            id: this.generateId(),
            timestamp: Date.now(),
            type: 'assistant_message',
            data: {
              content: payload.reply || payload.response || payload.content || 'Roboto responded.'
            }
          }
        ];

    for (const rawEvent of rawEvents) {
      yield this.normalizeEvent(rawEvent);
    }
  }

  async testBackend(): Promise<boolean> {
    return this.safePing(`${this.config.backendBaseUrl.replace(/\/$/, '')}/api/health`);
  }

  private async safePing(url: string): Promise<boolean> {
    try {
      const response = await fetch(url, { method: 'GET', headers: this.headers });
      return response.ok;
    } catch {
      return false;
    }
  }
}