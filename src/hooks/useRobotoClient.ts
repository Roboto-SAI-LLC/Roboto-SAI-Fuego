/**
 * React hook for RobotoClient SDK
 */

import { useState, useEffect, useCallback } from 'react';
import { RobotoClient, ChatEvent, RobotoClientConfig } from '../../../sdk/src';

const ROBOTO_CONFIG: RobotoClientConfig = {
  backendBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000',
  osAgentBaseUrl: import.meta.env.VITE_OS_AGENT_URL || 'http://localhost:5055'
};

export function useRobotoClient() {
  const [client] = useState(() => new RobotoClient(ROBOTO_CONFIG));
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  const testConnections = useCallback(async () => {
    setIsConnecting(true);
    try {
      const [backendOk, osAgentOk] = await Promise.all([
        client.testBackend(),
        client.testOsAgent()
      ]);
      setIsConnected(backendOk); // At least backend should be connected
      console.log('Connection test:', { backend: backendOk, osAgent: osAgentOk });
    } catch (error) {
      console.error('Connection test failed:', error);
      setIsConnected(false);
    } finally {
      setIsConnecting(false);
    }
  }, [client]);

  useEffect(() => {
    testConnections();
  }, [testConnections]);

  const sendMessage = useCallback(async (
    message: string,
    sessionId = 'default'
  ): Promise<ChatEvent[]> => {
    return client.chat(message, sessionId);
  }, [client]);

  const callTool = useCallback(async (
    source: 'backend' | 'mcp',
    toolName: string,
    args: Record<string, any>,
    serverId?: string
  ): Promise<any> => {
    if (source === 'backend') {
      return client.callBackendTool(toolName, args);
    } else {
      if (!serverId) throw new Error('serverId required for MCP tools');
      return client.callMcpTool(serverId, toolName, args);
    }
  }, [client]);

  return {
    client,
    isConnected,
    isConnecting,
    sendMessage,
    callTool,
    testConnections
  };
}