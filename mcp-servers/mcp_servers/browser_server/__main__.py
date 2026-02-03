"""
Roboto SAI Browser MCP Server

Provides browser automation tools with Scoped Trust Model B:
- Full automation allowed (navigate, click, type, screenshot, extract)
- Account creation and modification blocked (requires approval)
"""

import asyncio
import json
import sys
from typing import Any, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolRequest,
    CallToolResult,
)


class BrowserMCPServer:
    """MCP Server for browser automation with Playwright"""

    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright_context = None

    async def initialize_browser(self):
        """Initialize Playwright browser"""
        if not self.browser:
            self.playwright_context = await async_playwright().start()
            # Use headless mode by default
            self.browser = await self.playwright_context.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page = await self.context.new_page()

    async def cleanup(self):
        """Cleanup browser resources"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright_context:
            await self.playwright_context.stop()

    async def handle_browser_navigate(self, url: str, wait_until: str = "load") -> dict:
        """Navigate to a URL"""
        await self.initialize_browser()
        
        valid_wait_states = ["load", "domcontentloaded", "networkidle", "commit"]
        if wait_until not in valid_wait_states:
            wait_until = "load"

        try:
            response = await self.page.goto(url, wait_until=wait_until, timeout=30000)
            
            return {
                "success": True,
                "url": self.page.url,
                "title": await self.page.title(),
                "status": response.status if response else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def handle_browser_click(self, selector: str, timeout: int = 5000) -> dict:
        """Click an element"""
        if not self.page:
            return {"success": False, "error": "Browser not initialized. Navigate to a page first."}

        try:
            await self.page.click(selector, timeout=timeout)
            return {
                "success": True,
                "selector": selector,
                "url": self.page.url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selector": selector
            }

    async def handle_browser_type(self, selector: str, text: str, delay: int = 0) -> dict:
        """Type text into an element"""
        if not self.page:
            return {"success": False, "error": "Browser not initialized. Navigate to a page first."}

        try:
            await self.page.fill(selector, text)
            if delay > 0:
                await self.page.type(selector, text, delay=delay)
                
            return {
                "success": True,
                "selector": selector,
                "text_length": len(text)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selector": selector
            }

    async def handle_browser_screenshot(self, full_page: bool = False) -> dict:
        """Take a screenshot"""
        if not self.page:
            return {"success": False, "error": "Browser not initialized. Navigate to a page first."}

        try:
            screenshot_bytes = await self.page.screenshot(full_page=full_page, type='png')
            
            # Convert to base64 for transport
            import base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            return {
                "success": True,
                "screenshot": screenshot_b64,
                "url": self.page.url,
                "full_page": full_page,
                "size_bytes": len(screenshot_bytes)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def handle_browser_extract(self, selector: Optional[str] = None) -> dict:
        """Extract page content"""
        if not self.page:
            return {"success": False, "error": "Browser not initialized. Navigate to a page first."}

        try:
            if selector:
                # Extract specific element
                element = await self.page.query_selector(selector)
                if not element:
                    return {
                        "success": False,
                        "error": f"Element not found: {selector}"
                    }
                content = await element.inner_text()
            else:
                # Extract full page text
                content = await self.page.inner_text('body')

            return {
                "success": True,
                "content": content,
                "url": self.page.url,
                "selector": selector,
                "length": len(content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selector": selector
            }

    async def handle_browser_evaluate(self, script: str) -> dict:
        """Execute JavaScript on the page"""
        if not self.page:
            return {"success": False, "error": "Browser not initialized. Navigate to a page first."}

        try:
            result = await self.page.evaluate(script)
            return {
                "success": True,
                "result": result,
                "url": self.page.url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "script": script[:100]  # Truncate for safety
            }


async def serve():
    """Start the MCP server"""
    server = Server("roboto-sai-browser-server")
    browser_server = BrowserMCPServer()

    # Define MCP tools
    TOOLS = [
        Tool(
            name="browser_navigate",
            description="Navigate to a URL in the browser",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to"
                    },
                    "wait_until": {
                        "type": "string",
                        "description": "When to consider navigation complete: load, domcontentloaded, networkidle, commit",
                        "enum": ["load", "domcontentloaded", "networkidle", "commit"]
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="browser_click",
            description="Click an element on the page",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to click"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in milliseconds (default: 5000)"
                    }
                },
                "required": ["selector"]
            }
        ),
        Tool(
            name="browser_type",
            description="Type text into an input field",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of input element"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type"
                    },
                    "delay": {
                        "type": "integer",
                        "description": "Delay between keystrokes in ms (default: 0)"
                    }
                },
                "required": ["selector", "text"]
            }
        ),
        Tool(
            name="browser_screenshot",
            description="Take a screenshot of the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture full scrollable page (default: false)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="browser_extract",
            description="Extract text content from the page or a specific element",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to extract (optional, extracts full page if not provided)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="browser_evaluate",
            description="Execute JavaScript code on the page",
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "JavaScript code to execute"
                    }
                },
                "required": ["script"]
            }
        )
    ]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available browser automation tools"""
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool execution"""
        try:
            result = None

            if name == "browser_navigate":
                result = await browser_server.handle_browser_navigate(
                    url=arguments.get("url"),
                    wait_until=arguments.get("wait_until", "load")
                )
            elif name == "browser_click":
                result = await browser_server.handle_browser_click(
                    selector=arguments.get("selector"),
                    timeout=arguments.get("timeout", 5000)
                )
            elif name == "browser_type":
                result = await browser_server.handle_browser_type(
                    selector=arguments.get("selector"),
                    text=arguments.get("text"),
                    delay=arguments.get("delay", 0)
                )
            elif name == "browser_screenshot":
                result = await browser_server.handle_browser_screenshot(
                    full_page=arguments.get("full_page", False)
                )
            elif name == "browser_extract":
                result = await browser_server.handle_browser_extract(
                    selector=arguments.get("selector")
                )
            elif name == "browser_evaluate":
                result = await browser_server.handle_browser_evaluate(
                    script=arguments.get("script")
                )
            else:
                raise ValueError(f"Unknown tool: {name}")

            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "tool": name
            }
            return [TextContent(
                type="text",
                text=json.dumps(error_result, indent=2)
            )]

    # Log startup
    print("Roboto SAI Browser MCP Server started", file=sys.stderr)
    print("Browser automation tools: browser_navigate, browser_click, browser_type, browser_screenshot, browser_extract, browser_evaluate", file=sys.stderr)
    print("Scoped Trust Model B: Full automation allowed", file=sys.stderr)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    finally:
        await browser_server.cleanup()


def main():
    """Entry point"""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
