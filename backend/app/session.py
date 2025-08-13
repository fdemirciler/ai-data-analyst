from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import uuid
import threading
import time


@dataclass
class Session:
    session_id: str
    file_id: str
    created_at: float
    payload: Dict[str, Any]
    data_path: Path
    messages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)


class SessionStore:
    """In-memory session store. Not persistent. Single-file per session.

    Cleanup strategy: lazy eviction on lookup if TTL exceeded (optional later).
    """

    def __init__(self, ttl_seconds: int | None = None):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create(self, payload: Dict[str, Any], data_path: Path) -> Session:
        now = time.time()
        session_id = str(uuid.uuid4())
        file_id = session_id  # simple 1:1 mapping for MVP
        session = Session(
            session_id=session_id,
            file_id=file_id,
            created_at=now,
            payload=payload,
            data_path=data_path,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            if self._ttl is not None and (time.time() - session.created_at) > self._ttl:
                # Expire
                del self._sessions[session_id]
                return None
            return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        session = self.get(session_id)
        if not session:
            return
        message = {"role": role, "content": content}
        if extra:
            message.update(extra)
        session.messages.append(message)

    def replace_dataset(
        self, session_id: str, payload: Dict[str, Any], data_path: Path
    ) -> Session | None:
        session = self.get(session_id)
        if not session:
            return None
        session.payload = payload
        session.data_path = data_path
        return session

    def add_artifact(self, session_id: str, artifact: Dict[str, Any]) -> int:
        """Store an execution artifact (generated code, results, etc.)

        Returns index of the stored artifact for reference.
        """
        session = self.get(session_id)
        if not session:
            return -1
        session.artifacts.append(artifact)
        return len(session.artifacts) - 1


session_store = SessionStore()
