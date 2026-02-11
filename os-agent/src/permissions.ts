/**
 * Scoped Trust Model B - Permission Middleware for MCP OS Agent
 * 
 * Enforces permission policies for MCP tool calls based on:
 * - Tool name (filesystem, browser, twitter, email)
 * - Parameters (paths, actions, content)
 * - Risk level (LOW, MEDIUM, HIGH)
 * - Auto-approval settings
 */

import { z } from 'zod';
import winston from 'winston';

// Initialize logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'permissions.log' })
  ]
});

/**
 * Risk levels for tool operations
 */
export enum RiskLevel {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH'
}

/**
 * Permission check result
 */
export interface PermissionCheckResult {
  allowed: boolean;
  riskLevel: RiskLevel;
  reason: string;
  requiresApproval: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * Tool call parameters schema
 */
const ToolCallSchema = z.object({
  toolName: z.string(),
  parameters: z.record(z.unknown()),
  userId: z.string().optional(),
  sessionId: z.string().optional()
});

export type ToolCall = z.infer<typeof ToolCallSchema>;

/**
 * Scoped Trust Model B Configuration
 */
interface ScopedTrustConfig {
  filesystem: {
    allowedDrives: string[];
    allowRead: boolean;
    allowWrite: boolean;
  };
  browser: {
    allowFullAutomation: boolean;
  };
  twitter: {
    allowPosting: boolean;
    allowAccountCreation: boolean;
  };
  email: {
    allowSending: boolean;
    requireNotification: boolean;
  };
  autoApproveLowRisk: boolean;
}

const DEFAULT_CONFIG: ScopedTrustConfig = {
  filesystem: {
    allowedDrives: ['R:', 'D:'],
    allowRead: true,
    allowWrite: true
  },
  browser: {
    allowFullAutomation: true
  },
  twitter: {
    allowPosting: true,
    allowAccountCreation: false  // HIGH risk, requires approval
  },
  email: {
    allowSending: true,
    requireNotification: true
  },
  autoApproveLowRisk: true
};

/**
 * PermissionMiddleware - Enforces Scoped Trust Model B
 */
export class PermissionMiddleware {
  private config: ScopedTrustConfig;

  constructor(config: Partial<ScopedTrustConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('Permission middleware initialized with Scoped Trust Model B', { config: this.config });
  }

  /**
   * Check if a tool call is permitted
   */
  async checkPermission(toolCall: ToolCall): Promise<PermissionCheckResult> {
    // Validate input
    const parsed = ToolCallSchema.safeParse(toolCall);
    if (!parsed.success) {
      logger.error('Invalid tool call schema', { error: parsed.error, toolCall });
      return {
        allowed: false,
        riskLevel: RiskLevel.HIGH,
        reason: 'Invalid tool call parameters',
        requiresApproval: true
      };
    }

    const { toolName, parameters } = parsed.data;

    logger.info('Permission check requested', { toolName, parameters });

    // Route to specific permission checker based on tool category
    if (toolName.startsWith('fs_') || toolName.startsWith('file')) {
      return this.checkFilesystemPermission(toolName, parameters);
    } else if (toolName.startsWith('browser_') || toolName.startsWith('web_')) {
      return this.checkBrowserPermission(toolName, parameters);
    } else if (toolName.startsWith('twitter_')) {
      return this.checkTwitterPermission(toolName, parameters);
    } else if (toolName.startsWith('email_')) {
      return this.checkEmailPermission(toolName, parameters);
    }

    // Unknown tool - default to requiring approval
    logger.warn('Unknown tool category, defaulting to HIGH risk', { toolName });
    return {
      allowed: false,
      riskLevel: RiskLevel.HIGH,
      reason: `Unknown tool category: ${toolName}`,
      requiresApproval: true
    };
  }

  /**
   * Check filesystem tool permissions
   * Scoped Trust Model B: R:/D:/ read/write allowed
   */
  private checkFilesystemPermission(
    toolName: string,
    parameters: Record<string, unknown>
  ): PermissionCheckResult {
    const path = parameters.path as string;

    if (!path) {
      return {
        allowed: false,
        riskLevel: RiskLevel.HIGH,
        reason: 'No path specified for filesystem operation',
        requiresApproval: true
      };
    }

    // Extract drive letter
    const driveMatch = path.match(/^([A-Z]:)/);
    const drive = driveMatch ? driveMatch[1] : null;

    // Check if drive is allowed
    if (!drive || !this.config.filesystem.allowedDrives.includes(drive)) {
      logger.warn('Filesystem access denied: drive not allowed', { path, drive, allowedDrives: this.config.filesystem.allowedDrives });
      return {
        allowed: false,
        riskLevel: RiskLevel.HIGH,
        reason: `Access denied: ${drive || 'unknown drive'} not in allowed list (${this.config.filesystem.allowedDrives.join(', ')})`,
        requiresApproval: true,
        metadata: { path, drive, allowedDrives: this.config.filesystem.allowedDrives }
      };
    }

    // Determine operation type
    const isWrite = toolName.includes('write') || toolName.includes('create') || toolName.includes('delete') || toolName.includes('modify');
    const isRead = toolName.includes('read') || toolName.includes('list') || toolName.includes('search');

    // Check read permission
    if (isRead && !this.config.filesystem.allowRead) {
      return {
        allowed: false,
        riskLevel: RiskLevel.MEDIUM,
        reason: 'Filesystem read operations not permitted',
        requiresApproval: true,
        metadata: { path, operation: 'read' }
      };
    }

    // Check write permission
    if (isWrite && !this.config.filesystem.allowWrite) {
      return {
        allowed: false,
        riskLevel: RiskLevel.MEDIUM,
        reason: 'Filesystem write operations not permitted',
        requiresApproval: true,
        metadata: { path, operation: 'write' }
      };
    }

    // Determine risk level
    let riskLevel = RiskLevel.LOW;
    if (isWrite) {
      riskLevel = RiskLevel.MEDIUM;
    }

    logger.info('Filesystem permission granted', { toolName, path, riskLevel });

    return {
      allowed: true,
      riskLevel,
      reason: `Filesystem access permitted on ${drive}`,
      requiresApproval: !this.config.autoApproveLowRisk && riskLevel === RiskLevel.LOW ? false : riskLevel !== RiskLevel.LOW,
      metadata: { path, drive, operation: isWrite ? 'write' : 'read' }
    };
  }

  /**
   * Check browser tool permissions
   * Scoped Trust Model B: Full automation allowed
   */
  private checkBrowserPermission(
    toolName: string,
    parameters: Record<string, unknown>
  ): PermissionCheckResult {
    if (!this.config.browser.allowFullAutomation) {
      return {
        allowed: false,
        riskLevel: RiskLevel.HIGH,
        reason: 'Browser automation not permitted',
        requiresApproval: true
      };
    }

    // Browser automation is allowed but classify by risk
    const riskLevel = this.assessBrowserRisk(toolName, parameters);

    logger.info('Browser permission granted', { toolName, riskLevel });

    return {
      allowed: true,
      riskLevel,
      reason: 'Browser automation permitted',
      requiresApproval: riskLevel === RiskLevel.HIGH,
      metadata: { url: parameters.url, action: parameters.action }
    };
  }

  /**
   * Assess risk level for browser operations
   */
  private assessBrowserRisk(toolName: string, _parameters: Record<string, unknown>): RiskLevel {
    // High-risk browser operations
    if (
      toolName.includes('submit') ||
      toolName.includes('purchase') ||
      toolName.includes('payment') ||
      toolName.includes('delete_account')
    ) {
      return RiskLevel.HIGH;
    }

    // Medium-risk operations
    if (
      toolName.includes('login') ||
      toolName.includes('signup') ||
      toolName.includes('post') ||
      toolName.includes('send')
    ) {
      return RiskLevel.MEDIUM;
    }

    // Low-risk operations (reading, scrolling, clicking)
    return RiskLevel.LOW;
  }

  /**
   * Check Twitter tool permissions
   * Scoped Trust Model B: Posting allowed, account creation restricted (HIGH risk)
   */
  private checkTwitterPermission(
    toolName: string,
    parameters: Record<string, unknown>
  ): PermissionCheckResult {
    // Account creation is HIGH risk
    if (toolName.includes('create_account') || toolName.includes('signup')) {
      logger.warn('Twitter account creation requires approval', { toolName });
      return {
        allowed: this.config.twitter.allowAccountCreation,
        riskLevel: RiskLevel.HIGH,
        reason: 'Twitter account creation requires explicit approval',
        requiresApproval: true,
        metadata: { toolName, parameters }
      };
    }

    // Posting is allowed
    if (toolName.includes('post') || toolName.includes('tweet')) {
      if (!this.config.twitter.allowPosting) {
        return {
          allowed: false,
          riskLevel: RiskLevel.MEDIUM,
          reason: 'Twitter posting not permitted',
          requiresApproval: true
        };
      }

      logger.info('Twitter posting permitted', { toolName });
      return {
        allowed: true,
        riskLevel: RiskLevel.MEDIUM,
        reason: 'Twitter posting permitted',
        requiresApproval: false,
        metadata: { content: parameters.content }
      };
    }

    // Low-risk operations (scrolling, reading)
    logger.info('Twitter read operation permitted', { toolName });
    return {
      allowed: true,
      riskLevel: RiskLevel.LOW,
      reason: 'Twitter read operation permitted',
      requiresApproval: false
    };
  }

  /**
   * Check email tool permissions
   * Scoped Trust Model B: Sending allowed with notification
   */
  private checkEmailPermission(
    toolName: string,
    parameters: Record<string, unknown>
  ): PermissionCheckResult {
    if (!this.config.email.allowSending) {
      return {
        allowed: false,
        riskLevel: RiskLevel.MEDIUM,
        reason: 'Email sending not permitted',
        requiresApproval: true
      };
    }

    logger.info('Email sending permitted', { toolName, notification: this.config.email.requireNotification });

    return {
      allowed: true,
      riskLevel: RiskLevel.MEDIUM,
      reason: 'Email sending permitted' + (this.config.email.requireNotification ? ' (notification required)' : ''),
      requiresApproval: false,
      metadata: {
        to: parameters.to,
        subject: parameters.subject,
        requireNotification: this.config.email.requireNotification
      }
    };
  }

  /**
   * Log permission decision
   */
  logPermissionDecision(
    toolCall: ToolCall,
    result: PermissionCheckResult,
    action: 'approved' | 'denied' | 'requires_approval'
  ): void {
    logger.info('Permission decision', {
      ...toolCall,
      result,
      action,
      timestamp: new Date().toISOString()
    });
  }
}
