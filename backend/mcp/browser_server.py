"""Browser MCP Server for Twitter automation and web actions."""

import asyncio
from typing import Any, Sequence
from playwright.async_api import async_playwright, Browser, Page, Playwright
import mcp
from mcp.server import Server
from mcp.types import TextContent, PromptMessage

# Initialize MCP server
server = Server("browser-server")

# Global browser instance
_browser: Browser | None = None
_playwright: Playwright | None = None

async def get_browser() -> Browser:
    """Get or create browser instance."""
    global _browser, _playwright

    if _browser is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=False)  # Headless=False for visible actions

    return _browser

async def create_twitter_account(email: str, username: str, password: str) -> str:
    """Create a new Twitter account."""
    browser = await get_browser()
    page = await browser.new_page()

    try:
        # Navigate to Twitter signup
        await page.goto("https://twitter.com/i/flow/signup")

        # Fill signup form (simplified)
        await page.fill('input[placeholder="Phone or email"]', email)
        await page.fill('input[placeholder="Username"]', username)
        await page.fill('input[placeholder="Password"]', password)

        # Click signup button
        await page.click('button[data-testid="signup-button"]')

        # Wait for account creation
        await page.wait_for_timeout(5000)

        return f"Twitter account created for {username}"

    except Exception as e:
        return f"Failed to create Twitter account: {str(e)}"

    finally:
        await page.close()

async def scroll_twitter_feed(scroll_count: int = 5) -> str:
    """Scroll through Twitter feed."""
    browser = await get_browser()
    page = await browser.new_page()

    try:
        await page.goto("https://twitter.com/home")

        scrolled = 0
        for i in range(scroll_count):
            # Scroll down
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            scrolled += 1

        return f"Scrolled Twitter feed {scrolled} times"

    except Exception as e:
        return f"Failed to scroll Twitter feed: {str(e)}"

    finally:
        await page.close()

async def click_tweet(tweet_text: str) -> str:
    """Click on a tweet containing specific text."""
    browser = await get_browser()
    page = await browser.new_page()

    try:
        await page.goto("https://twitter.com/home")

        # Find tweet by text
        tweet_selector = f'div[data-testid="tweetText"]:has-text("{tweet_text}")'
        await page.wait_for_selector(tweet_selector, timeout=10000)
        await page.click(tweet_selector)

        return f"Clicked tweet containing: {tweet_text}"

    except Exception as e:
        return f"Failed to click tweet: {str(e)}"

    finally:
        await page.close()

async def type_tweet_text(text: str) -> str:
    """Type text into tweet composer."""
    browser = await get_browser()
    page = await browser.new_page()

    try:
        await page.goto("https://twitter.com/home")

        # Click compose button
        await page.click('a[data-testid="SideNav_NewTweet_Button"]')

        # Type text
        await page.fill('div[data-testid="tweetTextarea_0"]', text)

        return f"Typed tweet text: {text}"

    except Exception as e:
        return f"Failed to type tweet text: {str(e)}"

    finally:
        await page.close()

@server.list_tools()
async def handle_list_tools() -> list[mcp.types.Tool]:
    """List available tools."""
    return [
        mcp.types.Tool(
            name="create_twitter_account",
            description="Create a new Twitter account",
            input_schema={
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "username": {"type": "string"},
                    "password": {"type": "string"}
                },
                "required": ["email", "username", "password"]
            }
        ),
        mcp.types.Tool(
            name="scroll_twitter_feed",
            description="Scroll through Twitter feed",
            input_schema={
                "type": "object",
                "properties": {
                    "scroll_count": {"type": "integer", "default": 5}
                }
            }
        ),
        mcp.types.Tool(
            name="click_tweet",
            description="Click on a tweet containing specific text",
            input_schema={
                "type": "object",
                "properties": {
                    "tweet_text": {"type": "string"}
                },
                "required": ["tweet_text"]
            }
        ),
        mcp.types.Tool(
            name="type_tweet_text",
            description="Type text into tweet composer",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[mcp.types.TextContent]:
    """Handle tool calls."""
    if name == "create_twitter_account":
        result = await create_twitter_account(
            arguments["email"],
            arguments["username"],
            arguments["password"]
        )
    elif name == "scroll_twitter_feed":
        result = await scroll_twitter_feed(arguments.get("scroll_count", 5))
    elif name == "click_tweet":
        result = await click_tweet(arguments["tweet_text"])
    elif name == "type_tweet_text":
        result = await type_tweet_text(arguments["text"])
    else:
        result = f"Unknown tool: {name}"

    return [TextContent(type="text", text=result)]

async def main():
    """Main server entry point."""
    # Import here to avoid circular imports
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())