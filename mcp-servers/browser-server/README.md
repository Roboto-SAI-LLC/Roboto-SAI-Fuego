# Roboto SAI Browser MCP Server

Browser automation MCP server using Playwright with Scoped Trust Model B enforcement.

## Features

### Allowed Operations (Full Automation)
- **browser_navigate**: Navigate to any URL
- **browser_click**: Click elements on the page
- **browser_type**: Type text into input fields
- **browser_screenshot**: Capture page screenshots
- **browser_extract**: Extract text content from page
- **browser_evaluate**: Execute JavaScript code

### Scoped Trust Model B Compliance
- ✅ Full browser automation allowed
- ⚠️ Account creation/modification tools blocked (require approval)
- 🔒 Runs in headless mode by default for security

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install chromium
```

## Usage

The server communicates via stdio and is designed to be spawned by the OS Agent:

```bash
# Run standalone (for testing)
python server.py

# OS Agent spawns automatically:
# command: python
# args: ['-m', 'mcp_servers.browser_server']
```

## Tool Schemas

### browser_navigate
```json
{
  "url": "https://example.com",
  "wait_until": "load"  // load | domcontentloaded | networkidle | commit
}
```

### browser_click
```json
{
  "selector": "button.submit",
  "timeout": 5000
}
```

### browser_type
```json
{
  "selector": "input[name='search']",
  "text": "query text",
  "delay": 0  // milliseconds between keystrokes
}
```

### browser_screenshot
```json
{
  "full_page": false  // capture full scrollable page
}
```
Returns base64-encoded PNG image.

### browser_extract
```json
{
  "selector": ".content"  // optional, extracts full page if not provided
}
```

### browser_evaluate
```json
{
  "script": "document.title"
}
```

## Security

- Headless browser mode (no UI)
- No persistent browser data
- No extensions or plugins
- Sandboxed execution environment
- Path and permission validation via OS Agent

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Check tool registration
python server.py  # Should output available tools to stderr
```

## Architecture

```
OS Agent → stdio → Browser MCP Server → Playwright → Chromium
```

1. OS Agent spawns server as child process
2. Communication via stdio JSON-RPC
3. Playwright controls headless Chromium
4. Results returned as JSON via stdout

## Troubleshooting

**Browser fails to launch:**
```bash
# Reinstall Playwright browsers
python -m playwright install --force chromium
```

**Permission denied errors:**
- Ensure OS Agent permissions middleware allows browser tools
- Check tool names start with `browser_` prefix
- Verify Scoped Trust Model B configuration

**Timeout errors:**
- Increase timeout parameter in tool calls
- Check network connectivity
- Verify target page loads correctly

## Notes

- Browser state persists across tool calls within same session
- Navigate to a page before using click/type/extract tools
- Screenshots are base64-encoded in JSON response
- JavaScript execution is sandboxed to page context
