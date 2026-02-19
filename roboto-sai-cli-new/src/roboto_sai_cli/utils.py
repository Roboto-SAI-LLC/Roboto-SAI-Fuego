"""Utility functions for Roboto SAI CLI.

Why: Centralize reusable logic to maintain DRY principle and ensure consistent behavior across modules.
"""

import os
import secrets
from pathlib import Path
from typing import Any, Dict

import structlog
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field


# Global logger configuration
logger = structlog.get_logger(__name__)


class RoboConfig(BaseModel):
    """Configuration model for Roboto SAI CLI."""

    sigil: int = 929
    home: Path = Field(default_factory=lambda: Path.home() / ".roboto-sai")
    quantum_mock: bool = True
    memory_path: Path = Field(default_factory=lambda: Path.home() / ".roboto-sai" / "memory.json")
    encryption_key: bytes = Field(default_factory=Fernet.generate_key)

    def __init__(self, **data):
        super().__init__(**data)
        self.home.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.home / "memory.json"
        if not hasattr(self, 'encryption_key') or not self.encryption_key:
            self.encryption_key = Fernet.generate_key()


def get_config() -> RoboConfig:
    """Get application configuration with environment overrides.

    Why: Single source of truth for config to avoid scattered env checks.
    """
    config = RoboConfig()
    config.sigil = int(os.getenv("SIGIL", str(config.sigil)))
    config.quantum_mock = os.getenv("ROBOTO_QUANTUM_MOCK", str(config.quantum_mock)).lower() == "true"
    home_override = os.getenv("ROBOTO_HOME")
    if home_override:
        config.home = Path(home_override)
        config.memory_path = config.home / "memory.json"
    return config


def secure_random_hex(length: int = 32) -> str:
    """Generate cryptographically secure random hex.

    Why: Use secrets module for true randomness in security-critical operations.
    """
    return secrets.token_hex(length // 2 + length % 2)[:length]


def encrypt_data(data: str, key: bytes) -> str:
    """Encrypt string data using Fernet.

    Why: Secure sensitive data at rest with industry-standard encryption.
    """
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()


def decrypt_data(encrypted: str, key: bytes) -> str:
    """Decrypt string data using Fernet.

    Why: Paired with encrypt_data for secure data retrieval.
    """
    f = Fernet(key)
    return f.decrypt(encrypted.encode()).decode()


def self_critique(func_name: str, performance: float, security: int, readability: int, fidelity: int) -> str:
    """Internal self-critique for code quality.

    Why: Automated improvement suggestions to maintain god-tier standards.
    """
    suggestion = ""
    if performance < 0.8:
        suggestion += "Consider async optimizations or caching. "
    if security < 8:
        suggestion += "Add input validation/sanitization. "
    if readability < 7:
        suggestion += "Add type hints and docstrings. "
    if fidelity < 9:
        suggestion += "Review quantum logic against Qiskit best practices. "
    return suggestion.strip() or "Excellent! No improvements needed."


# Rate limiter for anti-abuse
class RateLimiter:
    """In-memory rate limiter with TTL.

    Why: Prevent abuse without external dependencies, using dict-based storage.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests: Dict[str, list] = {}
        self.limit = requests_per_minute

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for the key.

        Why: Simple sliding window implementation for rate limiting.
        """
        now = __import__('time').time()
        if key not in self.requests:
            self.requests[key] = []

        # Clean old requests
        self.requests[key] = [t for t in self.requests[key] if now - t < 60]

        if len(self.requests[key]) >= self.limit:
            return False

        self.requests[key].append(now)
        return True