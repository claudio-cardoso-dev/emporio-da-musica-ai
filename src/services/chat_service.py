"""Conversation session orchestration and SQLite persistence service."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.assistant_agent import AssistantAgent
from src.db.database import db_cursor
from src.db.ingestion import ingest_all_csvs


class ChatService:
    """
    Manages multi-turn conversational sessions, persists chat history in SQLite,
    and delegates message processing to the AssistantAgent.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        pdf_path: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        # Ensure operational tables and CSV data are initialized
        ingest_all_csvs(db_path=self.db_path, force_reload=False)
        self.agent = AssistantAgent(
            db_path=self.db_path,
            pdf_path=pdf_path,
            api_key=api_key,
        )

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a specific session_id in chronological order."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                SELECT id, session_id, role, content, metadata_json, created_at
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = list(reversed(cursor.fetchall()))
            history = []
            for row in rows:
                item = dict(row)
                try:
                    item["metadata"] = json.loads(item.get("metadata_json") or "{}")
                except json.JSONDecodeError:
                    item["metadata"] = {}
                item.pop("metadata_json", None)
                history.append(item)
            return history

    def _save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a single turn message into SQLite chat_history."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                INSERT INTO chat_history (session_id, role, content, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def clear_session(self, session_id: str) -> None:
        """Reset conversation history for a given session_id."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute("DELETE FROM chat_history WHERE session_id = ?;", (session_id,))
        if hasattr(self.agent, "clear_session_state"):
            self.agent.clear_session_state(session_id)

    def process_message(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        """Process a user message with full session history and persist both turns."""
        prior_history = self.get_history(session_id=session_id, limit=12)
        formatted_history = [
            {
                "role": h["role"],
                "content": h["content"],
                "metadata": h.get("metadata", {}),
            }
            for h in prior_history
        ]

        self._save_message(session_id=session_id, role="user", content=user_message)

        result = self.agent.run(
            user_message=user_message,
            session_id=session_id,
            chat_history=formatted_history,
        )

        self._save_message(
            session_id=session_id,
            role="assistant",
            content=result["reply"],
            metadata={
                "language": result.get("language"),
                "confidence": result.get("confidence"),
                "model_used": result.get("model_used"),
                "tools_called": result.get("tools_called", []),
                "active_agent": result.get("active_agent", "catalog"),
                "logged_failed_intent": result.get("logged_failed_intent", False),
            },
        )
        return result
