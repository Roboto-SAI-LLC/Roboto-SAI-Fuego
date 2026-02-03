/**
 * RobotoClient SDK
 * Main interface for interacting with Roboto SAI backend and MCP tools
 */

import { ChatEvent, ToolCall, RobotoClientConfig } from './types';

export class RobotoClient {
  constructor(private config: RobotoClientConfig) {}

  /**
   * Send a chat message and get response
   */
  async chat(message: string, sessionId = 'default'): Promise<ChatEvent[]> {
    const response = await fetch(`${this.config.backendBaseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.config.defaultHeaders
      },
      body: JSON.stringify({
        sessionId,
        message,
        timestamp: Date.now()
      })
    });

    if (!response.ok) {
      throw new Error(`Chat failed: ${response.status}`);
    }

    const data = await response.json();
    return [{
      type: 'assistant_message',
      timestamp: Date.now(),
      data: { content: data.reply || data.message || 'Response received' }
    }];
  }

  /**
   * Call a backend tool directly
   */
  async callBackendTool(toolName: string, args: Record<string, any>): Promise<any> {
    const response = await fetch(`${this.config.backendBaseUrl}/api/tools/${toolName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.config.defaultHeaders
      },
      body: JSON.stringify(args)
    });

    if (!response.ok) {
      throw new Error(`Tool call failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Call an MCP tool via OS agent
   */
  async callMcpTool(serverId: string, toolName: string, args: Record<string, any>): Promise<any> {
    const response = await fetch(`${this.config.osAgentBaseUrl}/mcp/call`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.config.defaultHeaders
      },
      body: JSON.stringify({
        serverId,
        toolName,
        args
      })
    });

    if (!response.ok) {
      throw new Error(`MCP tool call failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Test backend connection
   */
  async testBackend(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.backendBaseUrl}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Test OS agent connection
   */
  async testOsAgent(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.osAgentBaseUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}