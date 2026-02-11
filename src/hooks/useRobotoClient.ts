import { useCallback, useEffect, useState } from 'react';
import {
  ChatEventPayload,
  RobotoClient,
  RobotoClientConfig
} from '../../sdk/src';

const ROBOTO_CONFIG: RobotoClientConfig = {
  backendBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000',
};

interface ChatStreamRequest {
  message: string;
  sessionId?: string;
  context?: Record<string, unknown>;
  reasoningEffort?: string;
  userId?: string;
}

export function useRobotoClient() {
  const [client] = useState(() => new RobotoClient(ROBOTO_CONFIG));
  const [isConnected, setIsConnected] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [events, setEvents] = useState<ChatEventPayload[]>([]);

  const refreshConnections = useCallback(async () => {
    setIsChecking(true);
    try {
      const backendOk = await client.testBackend();
      setIsConnected(backendOk);
    } catch (error) {
      console.error('Connection test failed', error);
      setIsConnected(false);
    } finally {
      setIsChecking(false);
    }
  }, [client]);

  useEffect(() => {
    void refreshConnections();
  }, [refreshConnections]);

  const handleEvent = useCallback((event: ChatEventPayload) => {
    setEvents((prev) => [...prev, event].slice(-80));
  }, []);

  const sendMessage = useCallback(async (
    payload: ChatStreamRequest,
    onEvent?: (event: ChatEventPayload) => void
  ) => {
    try {
      for await (const event of client.streamChat(payload)) {
        handleEvent(event);
        onEvent?.(event);
      }
    } catch (error) {
      handleEvent({
        type: 'error',
        id: `err_${Date.now()}`,
        timestamp: Date.now(),
        data: {
          message: (error as Error).message,
          details: error
        }
      });
    }
  }, [client, handleEvent]);

  const clearEvents = useCallback(() => setEvents([]), []);

  return {
    client,
    isConnected,
    isChecking,
    events,
    refreshConnections,
    sendMessage,
    clearEvents
  };
}
