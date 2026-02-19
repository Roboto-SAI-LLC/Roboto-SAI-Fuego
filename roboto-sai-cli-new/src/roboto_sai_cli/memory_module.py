"""Memory management module for Roboto SAI CLI.

Why: Persistent logging of interactions and patterns for evolutionary improvement.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

from .utils import get_config, logger, encrypt_data, decrypt_data


class MemoryEntry(BaseModel):
    """Structured memory entry."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    category: str
    data: Dict
    confidence: str = Field(..., pattern="^(high|medium|low)$")
    quantum_valence: float = Field(..., ge=-1, le=1, description="Emotional/emotional quantum state")
    engagement_signal: str = Field(..., description="User engagement indicator")


class MemoryBank:
    """Encrypted persistent memory storage."""

    def __init__(self):
        self.config = get_config()
        self.entries: List[MemoryEntry] = []
        self._load()

    def log_entry(self, category: str, data: Dict, confidence: str = "medium",
                  valence: float = 0.0, engagement: str = "neutral") -> None:
        """Log a new memory entry.

        Why: Track all interactions for pattern analysis and improvement.
        """
        entry = MemoryEntry(
            category=category,
            data=data,
            confidence=confidence,
            quantum_valence=valence,
            engagement_signal=engagement
        )
        self.entries.append(entry)
        self._save()
        logger.info("Memory entry logged", category=category, confidence=confidence)

    def get_summary(self) -> Dict:
        """Get memory summary statistics.

        Why: Provide insights into usage patterns and emotional trends.
        """
        total = len(self.entries)
        if total == 0:
            return {"total_entries": 0}

        categories = {}
        valences = []
        confidences = {"high": 0, "medium": 0, "low": 0}

        for entry in self.entries:
            categories[entry.category] = categories.get(entry.category, 0) + 1
            valences.append(entry.quantum_valence)
            confidences[entry.confidence] += 1

        avg_valence = sum(valences) / len(valences)
        return {
            "total_entries": total,
            "categories": categories,
            "average_quantum_valence": avg_valence,
            "confidence_distribution": confidences,
            "recent_entry": self.entries[-1].timestamp.isoformat() if self.entries else None
        }

    def _load(self) -> None:
        """Load encrypted memory from disk.

        Why: Persistent storage with security through encryption.
        """
        if not self.config.memory_path.exists():
            return

        try:
            with open(self.config.memory_path, 'r') as f:
                encrypted = f.read().strip()
            if encrypted:
                decrypted = decrypt_data(encrypted, self.config.encryption_key)
                data = json.loads(decrypted)
                self.entries = [MemoryEntry(**item) for item in data]
        except Exception as e:
            logger.warning("Failed to load memory", error=str(e))

    def _save(self) -> None:
        """Save memory to encrypted disk.

        Why: Secure persistence of all logged data.
        """
        try:
            data = [entry.model_dump() for entry in self.entries]
            json_str = json.dumps(data, default=str, indent=2)
            encrypted = encrypt_data(json_str, self.config.encryption_key)
            with open(self.config.memory_path, 'w') as f:
                f.write(encrypted)
        except Exception as e:
            logger.error("Failed to save memory", error=str(e))


# Global memory bank instance
memory_bank = MemoryBank()