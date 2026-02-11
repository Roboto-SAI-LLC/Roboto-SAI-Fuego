/**
 * Roboto SAI Filesystem MCP Server
 * 
 * Provides file operation tools with Scoped Trust Model B path restrictions.
 * Only allows operations within R:/ and D:/ drives on Windows.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import fs from 'fs/promises';
import path from 'path';
import { constants as fsConstants } from 'fs';

/**
 * Path validation for Scoped Trust Model B
 * Only allows R:/ and D:/ drives
 */
function isPathAllowed(filePath: string): boolean {
  const normalized = path.normalize(filePath).toUpperCase();
  return normalized.startsWith('R:\\') || normalized.startsWith('D:\\');
}

/**
 * Validate and resolve path
 */
function validatePath(filePath: string): string {
  if (!filePath) {
    throw new Error('Path is required');
  }

  const resolved = path.resolve(filePath);
  
  if (!isPathAllowed(resolved)) {
    throw new Error(
      `Access denied: Path must be within R:/ or D:/ drives. Got: ${resolved}`
    );
  }

  return resolved;
}

/**
 * Check if path exists
 */
async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

/**
 * MCP Tool Definitions
 */
const TOOLS: Tool[] = [
  {
    name: 'fs_listDir',
    description: 'List contents of a directory (files and subdirectories)',
    inputSchema: {
      type: 'object' as const,
      properties: {
        path: {
          type: 'string',
          description: 'Directory path to list (must be within R:/ or D:/)',
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'fs_readFile',
    description: 'Read contents of a text file',
    inputSchema: {
      type: 'object' as const,
      properties: {
        path: {
          type: 'string',
          description: 'File path to read (must be within R:/ or D:/)',
        },
        encoding: {
          type: 'string',
          description: 'File encoding (default: utf-8)',
          enum: ['utf-8', 'ascii', 'utf-16le', 'base64'],
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'fs_writeFile',
    description: 'Write content to a file (creates or overwrites)',
    inputSchema: {
      type: 'object' as const,
      properties: {
        path: {
          type: 'string',
          description: 'File path to write (must be within R:/ or D:/)',
        },
        content: {
          type: 'string',
          description: 'Content to write to the file',
        },
        createDirs: {
          type: 'boolean',
          description: 'Create parent directories if they don\'t exist (default: false)',
        },
      },
      required: ['path', 'content'],
    },
  },
  {
    name: 'fs_searchInFiles',
    description: 'Search for text pattern in files within a directory',
    inputSchema: {
      type: 'object' as const,
      properties: {
        rootPath: {
          type: 'string',
          description: 'Root directory to search in (must be within R:/ or D:/)',
        },
        query: {
          type: 'string',
          description: 'Text pattern to search for (case-insensitive)',
        },
        filePattern: {
          type: 'string',
          description: 'File extension pattern (e.g., ".ts", ".md", "*") (default: "*")',
        },
        maxResults: {
          type: 'number',
          description: 'Maximum number of results to return (default: 50)',
        },
      },
      required: ['rootPath', 'query'],
    },
  },
];

/**
 * Tool handler implementations
 */

async function handleListDir(args: { path: string }): Promise<string> {
  const dirPath = validatePath(args.path);

  if (!(await pathExists(dirPath))) {
    throw new Error(`Directory not found: ${dirPath}`);
  }

  const stat = await fs.stat(dirPath);
  if (!stat.isDirectory()) {
    throw new Error(`Path is not a directory: ${dirPath}`);
  }

  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  
  const results = entries.map((entry) => ({
    name: entry.name,
    type: entry.isDirectory() ? 'directory' : 'file',
    path: path.join(dirPath, entry.name),
  }));

  return JSON.stringify({
    directory: dirPath,
    count: results.length,
    entries: results,
  }, null, 2);
}

async function handleReadFile(args: { path: string; encoding?: string }): Promise<string> {
  const filePath = validatePath(args.path);

  if (!(await pathExists(filePath))) {
    throw new Error(`File not found: ${filePath}`);
  }

  const stat = await fs.stat(filePath);
  if (!stat.isFile()) {
    throw new Error(`Path is not a file: ${filePath}`);
  }

  const encoding = (args.encoding || 'utf-8') as BufferEncoding;
  const content = await fs.readFile(filePath, encoding);

  return JSON.stringify({
    path: filePath,
    size: stat.size,
    encoding,
    content,
  }, null, 2);
}

async function handleWriteFile(
  args: { path: string; content: string; createDirs?: boolean }
): Promise<string> {
  const filePath = validatePath(args.path);
  const createDirs = args.createDirs ?? false;

  // Create parent directories if requested
  if (createDirs) {
    const dirPath = path.dirname(filePath);
    await fs.mkdir(dirPath, { recursive: true });
  }

  await fs.writeFile(filePath, args.content, 'utf-8');

  const stat = await fs.stat(filePath);

  return JSON.stringify({
    path: filePath,
    size: stat.size,
    written: true,
  }, null, 2);
}

async function handleSearchInFiles(
  args: { rootPath: string; query: string; filePattern?: string; maxResults?: number }
): Promise<string> {
  const rootPath = validatePath(args.rootPath);
  const query = args.query.toLowerCase();
  const filePattern = args.filePattern || '*';
  const maxResults = args.maxResults || 50;

  if (!(await pathExists(rootPath))) {
    throw new Error(`Directory not found: ${rootPath}`);
  }

  const results: Array<{ file: string; line: number; content: string }> = [];

  async function searchDir(dirPath: string): Promise<void> {
    if (results.length >= maxResults) return;

    const entries = await fs.readdir(dirPath, { withFileTypes: true });

    for (const entry of entries) {
      if (results.length >= maxResults) break;

      const fullPath = path.join(dirPath, entry.name);

      if (entry.isDirectory()) {
        // Recursively search subdirectories
        await searchDir(fullPath);
      } else if (entry.isFile()) {
        // Check file pattern
        if (filePattern !== '*' && !entry.name.endsWith(filePattern)) {
          continue;
        }

        try {
          const content = await fs.readFile(fullPath, 'utf-8');
          const lines = content.split('\n');

          for (let i = 0; i < lines.length; i++) {
            if (results.length >= maxResults) break;

            const line = lines[i];
            if (line.toLowerCase().includes(query)) {
              results.push({
                file: fullPath,
                line: i + 1,
                content: line.trim(),
              });
            }
          }
        } catch (error) {
          // Skip files that can't be read (binary, permissions, etc.)
          continue;
        }
      }
    }
  }

  await searchDir(rootPath);

  return JSON.stringify({
    query: args.query,
    rootPath,
    filePattern,
    resultsCount: results.length,
    results,
  }, null, 2);
}

/**
 * Main server initialization
 */
async function main() {
  // Create MCP server
  const server = new Server(
    {
      name: 'roboto-sai-fs-server',
      version: '1.0.0',
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Register list tools handler
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS,
  }));

  // Register call tool handler
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      let result: string;

      switch (name) {
        case 'fs_listDir':
          result = await handleListDir(args as { path: string });
          break;

        case 'fs_readFile':
          result = await handleReadFile(args as { path: string; encoding?: string });
          break;

        case 'fs_writeFile':
          result = await handleWriteFile(
            args as { path: string; content: string; createDirs?: boolean }
          );
          break;

        case 'fs_searchInFiles':
          result = await handleSearchInFiles(
            args as {
              rootPath: string;
              query: string;
              filePattern?: string;
              maxResults?: number;
            }
          );
          break;

        default:
          throw new Error(`Unknown tool: ${name}`);
      }

      return {
        content: [
          {
            type: 'text' as const,
            text: result,
          },
        ],
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);

      return {
        content: [
          {
            type: 'text' as const,
            text: JSON.stringify({
              error: errorMessage,
              tool: name,
            }, null, 2),
          },
        ],
        isError: true,
      };
    }
  });

  // Start server with stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('Roboto SAI Filesystem MCP Server started');
  console.error('Allowed drives: R:/, D:/');
  console.error('Tools: fs_listDir, fs_readFile, fs_writeFile, fs_searchInFiles');
}

// Start server
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
