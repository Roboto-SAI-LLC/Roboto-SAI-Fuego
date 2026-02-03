/**
 * RobotoClient SDK Types
 * Shared interfaces for chat, tools, and MCP integration
 */

export interface ChatEvent {
  type: 'assistant_message' | 'tool_call' | 'tool_result' | 'approval_required' | 'error';
  id?: string;
  timestamp: number;
  data: any;
}

export interface ToolCall {
  id: string;
  source: 'backend' | 'mcp';
  serverId?: string;
  toolName: string;
  args: Record<string, any>;
  description?: string;
}

export interface ApprovalRequest {
  id: string;
  description: string;
  riskLevel: 'low' | 'medium' | 'high';
  suggestedAction: 'approve' | 'deny' | 'modify';
  toolCall: ToolCall;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  context?: Record<string, any>;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface McpServer {
  id: string;
  name: string;
  description: string;
  tools: McpTool[];
  enabled: boolean;
}

export interface McpTool {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
}

export interface RobotoClientConfig {
  backendBaseUrl: string;
  osAgentBaseUrl: string;
  defaultHeaders?: Record<string, string>;
}