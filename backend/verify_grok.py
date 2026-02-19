"""Minimal Grok verification helper for local debugging.

- Prints request payload and response (masked where appropriate).
- Exits with non-zero code on HTTP error so startup can fail in local debug mode.

Usage: python verify_grok.py
Environment:
  XAI_API_KEY, XAI_MODEL, XAI_API_BASE_URL (optional)
"""
import os
import sys
import json
import logging
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_grok")


def main() -> int:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        logger.error("XAI_API_KEY not configured in environment")
        return 2

    model = os.getenv("XAI_MODEL", os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning"))
    base = (os.getenv("XAI_API_BASE_URL") or "https://api.x.ai").rstrip("/")
    url = f"{base}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a short Grok connectivity test."},
            {"role": "user", "content": "Say hello."},
        ],
        "temperature": 0,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    logger.info("Verifying Grok connectivity: url=%s model=%s", url, model)
    logger.debug("Payload: %s", json.dumps(payload))

    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=payload, headers=headers)
            logger.info("Grok verify status: %s", r.status_code)
            try:
                body: Any = r.json()
            except Exception:
                body = r.text
            logger.info("Grok verify body: %s", json.dumps(body) if isinstance(body, dict) else str(body))

            if r.status_code != 200:
                logger.error("Grok verify failed with status %s", r.status_code)
                return 1

        return 0

    except Exception as e:
        logger.exception("Grok verify unexpected error: %s", e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
