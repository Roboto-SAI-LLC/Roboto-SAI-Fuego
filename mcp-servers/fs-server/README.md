# Roboto SAI Filesystem MCP Server

MCP server providing file system operations with **Scoped Trust Model B** restrictions.

## Features

- **Path Restrictions**: Only allows operations within `R:/` and `D:/` drives
- **Four Core Tools**:
  - `listDir`: List directory contents
  - `readFile`: Read file content
  - `writeFile`: Write/create files
  - `searchInFiles`: Search for text patterns in files
- **Safety Features**:
  - Path validation on all operations
  - Error handling for missing files/directories
  - Automatic encoding support
  - Recursive directory creation option

## Tools

### listDir
Lists all files and subdirectories in a directory.

**Input:**
```json
{
  "path": "R:/Repos/MyProject"
}
```

**Output:**
```json
{
  "directory": "R:\\Repos\\MyProject",
  "count": 3,
  "entries": [
    { "name": "src", "type": "directory", "path": "R:\\Repos\\MyProject\\src" },
    { "name": "README.md", "type": "file", "path": "R:\\Repos\\MyProject\\README.md" },
    { "name": "package.json", "type": "file", "path": "R:\\Repos\\MyProject\\package.json" }
  ]
}
```

### readFile
Reads the contents of a text file.

**Input:**
```json
{
  "path": "R:/Repos/MyProject/README.md",
  "encoding": "utf-8"
}
```

**Output:**
```json
{
  "path": "R:\\Repos\\MyProject\\README.md",
  "size": 1234,
  "encoding": "utf-8",
  "content": "# My Project\n\nThis is my project..."
}
```

### writeFile
Writes content to a file (creates or overwrites).

**Input:**
```json
{
  "path": "R:/Repos/MyProject/notes.txt",
  "content": "My notes here",
  "createDirs": true
}
```

**Output:**
```json
{
  "path": "R:\\Repos\\MyProject\\notes.txt",
  "size": 14,
  "written": true
}
```

### searchInFiles
Searches for text patterns in files within a directory tree.

**Input:**
```json
{
  "rootPath": "R:/Repos/MyProject",
  "query": "import",
  "filePattern": ".ts",
  "maxResults": 50
}
```

**Output:**
```json
{
  "query": "import",
  "rootPath": "R:\\Repos\\MyProject",
  "filePattern": ".ts",
  "resultsCount": 15,
  "results": [
    {
      "file": "R:\\Repos\\MyProject\\src\\index.ts",
      "line": 1,
      "content": "import { Server } from '@modelcontextprotocol/sdk';"
    }
  ]
}
```

## Installation

```bash
cd mcp-servers/fs-server
npm install
npm run build
```

## Usage

### Standalone (stdio)
```bash
npm start
# Server listens on stdin/stdout for MCP JSON-RPC messages
```

### From OS Agent
The OS Agent automatically connects to this server via stdio transport.

Configuration in `os-agent/src/main.ts`:
```typescript
{
  name: 'filesystem',
  command: 'node',
  args: ['R:/Repos/Roboto-SAI-2026/mcp-servers/fs-server/dist/index.js'],
  env: {
    ALLOWED_DRIVES: 'R:,D:',
  },
  enabled: true
}
```

## Development

```bash
# Watch mode
npm run dev

# Type checking
npm run typecheck

# Build
npm run build
```

## Security

### Scoped Trust Model B
- **Filesystem Access**: R:/ and D:/ drives only
- **Read Operations**: LOW to MEDIUM risk
- **Write Operations**: MEDIUM risk (requires approval in OS Agent)
- **Path Validation**: All paths normalized and validated before operations
- **Error Handling**: Graceful failure with descriptive error messages

### Path Examples

**Allowed:**
- `R:/Repos/MyProject/file.txt`
- `D:/Documents/notes.md`
- `R:\Users\user\Desktop\code\app.js`

**Denied:**
- `C:/Windows/System32/config.sys` (C: drive)
- `E:/ExternalDrive/file.txt` (E: drive)
- `/etc/passwd` (Unix path)
- `\\\\network\\share\\file` (Network path)

## Error Handling

All tools return structured error responses:
```json
{
  "error": "Access denied: Path must be within R:/ or D:/ drives. Got: C:\\Windows",
  "tool": "readFile"
}
```

## License

MIT

## Author

Roberto Villarreal Martinez <ytkrobthugod@hotmail.com>
