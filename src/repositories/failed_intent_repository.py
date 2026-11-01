"""Repository for logging ambiguous or low-confidence user messages (inspired by XPTO case)."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.db.database import db_cursor


class FailedIntentRepository:
    """Persist ambiguous or low-confidence interactions for continuous NLU improvement."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path

    def log_failure(
        self,
        session_id: str,
        user_message: str,
        detected_language: str = "pt",
        reason: str = "low_confidence",
        confidence: float = 0.0,
        resolution_strategy: str = "ask_rephrase",
    ) -> int:
        """Record a failed or ambiguous intent event and return its row ID."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                INSERT INTO failed_intents (
                    session_id, user_message, detected_language,
                    reason, confidence, resolution_strategy
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_message,
                    detected_language,
                    reason,
                    confidence,
                    resolution_strategy,
                ),
            )
            return int(cursor.lastrowid)

    def get_all_failures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve logged failed/ambiguous intents for observability and evaluation."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                "SELECT * FROM failed_intents ORDER BY id DESC LIMIT ?;",
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]
