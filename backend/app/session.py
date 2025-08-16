from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import uuid
import threading
import time
import orjson
import redis

from .config import settings


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

    def message_count(self, session_id: str) -> int:
        session = self.get(session_id)
        return len(session.messages) if session else 0


class RedisSessionStore:
    """Redis-backed session store keeping API parity with SessionStore.

    Keys (with prefix):
      {prefix}:sess:{id}:meta       -> HSET fields: session_id, file_id, created_at, data_path, payload(JSON)
      {prefix}:sess:{id}:messages   -> RPUSH JSON message objects
      {prefix}:sess:{id}:artifacts  -> RPUSH JSON artifact objects

    TTL: Fixed from creation. On each push, align lists' TTL to meta's remaining TTL.
    """

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int | None, key_prefix: str = "ai-da"):
        self._r = redis_client
        self._ttl = ttl_seconds
        self._prefix = key_prefix.strip() if key_prefix else "ai-da"

    # Key helpers
    def _k_meta(self, sid: str) -> str:
        return f"{self._prefix}:sess:{sid}:meta"

    def _k_msgs(self, sid: str) -> str:
        return f"{self._prefix}:sess:{sid}:messages"

    def _k_art(self, sid: str) -> str:
        return f"{self._prefix}:sess:{sid}:artifacts"

    @staticmethod
    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")

    @staticmethod
    def _loads(s: str) -> Any:
        return orjson.loads(s)

    def _align_list_ttls(self, sid: str) -> None:
        meta = self._k_meta(sid)
        pttl = self._r.pttl(meta)
        # If meta has TTL, align lists
        if isinstance(pttl, int) and pttl > 0:
            pipe = self._r.pipeline()
            pipe.pexpire(self._k_msgs(sid), pttl)
            pipe.pexpire(self._k_art(sid), pttl)
            pipe.execute()

    def _refresh_all_ttls(self, sid: str) -> None:
        """Set a sliding TTL on meta/messages/artifacts to the full TTL, if configured."""
        if self._ttl and self._ttl > 0:
            ttl = int(self._ttl)
            pipe = self._r.pipeline()
            pipe.expire(self._k_meta(sid), ttl)
            pipe.expire(self._k_msgs(sid), ttl)
            pipe.expire(self._k_art(sid), ttl)
            pipe.execute()

    def create(self, payload: Dict[str, Any], data_path: Path) -> Session:
        now = time.time()
        session_id = str(uuid.uuid4())
        file_id = session_id
        meta_key = self._k_meta(session_id)
        pipe = self._r.pipeline()
        pipe.hset(
            meta_key,
            mapping={
                "session_id": session_id,
                "file_id": file_id,
                "created_at": str(now),
                "data_path": str(data_path),
                "payload": self._dumps(payload),
            },
        )
        if self._ttl and self._ttl > 0:
            pipe.expire(meta_key, int(self._ttl))
        pipe.execute()
        # Initialize TTLs on lists as well (sliding TTL maintained on writes)
        self._refresh_all_ttls(session_id)
        return Session(
            session_id=session_id,
            file_id=file_id,
            created_at=now,
            payload=payload,
            data_path=data_path,
        )

    def get(self, session_id: str) -> Session | None:
        meta_key = self._k_meta(session_id)
        meta = self._r.hgetall(meta_key)
        if not meta:
            return None
        try:
            created_at = float(meta.get("created_at", "0"))
            data_path = Path(meta.get("data_path", ""))
            payload = self._loads(meta.get("payload", "{}"))
        except Exception:
            return None

        # Load artifacts for compatibility with list_artifacts route
        arts_raw = self._r.lrange(self._k_art(session_id), 0, -1)
        artifacts: List[Dict[str, Any]] = []
        for a in arts_raw:
            try:
                artifacts.append(self._loads(a))
            except Exception:
                continue

        return Session(
            session_id=session_id,
            file_id=meta.get("file_id", session_id),
            created_at=created_at,
            payload=payload,
            data_path=data_path,
            messages=[],  # intentionally not loading messages list
            artifacts=artifacts,
        )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        msg = {"role": role, "content": content}
        if extra:
            msg.update(extra)
        key_msgs = self._k_msgs(session_id)
        max_keep = settings.redis_max_messages or 10
        pipe = self._r.pipeline()
        pipe.rpush(key_msgs, self._dumps(msg))
        # Trim to last N to avoid unbounded growth (no history needed)
        pipe.ltrim(key_msgs, -max_keep, -1)
        # Sliding TTL on all related keys
        if self._ttl and self._ttl > 0:
            ttl = int(self._ttl)
            pipe.expire(self._k_meta(session_id), ttl)
            pipe.expire(key_msgs, ttl)
            pipe.expire(self._k_art(session_id), ttl)
        pipe.execute()

    def replace_dataset(
        self, session_id: str, payload: Dict[str, Any], data_path: Path
    ) -> Session | None:
        meta_key = self._k_meta(session_id)
        if not self._r.exists(meta_key):
            return None
        mapping = {
            "payload": self._dumps(payload),
            "data_path": str(data_path),
        }
        self._r.hset(meta_key, mapping=mapping)
        return self.get(session_id)

    def add_artifact(self, session_id: str, artifact: Dict[str, Any]) -> int:
        key_art = self._k_art(session_id)
        max_keep = settings.redis_max_artifacts or 10
        pipe = self._r.pipeline()
        pipe.rpush(key_art, self._dumps(artifact))
        pipe.ltrim(key_art, -max_keep, -1)
        if self._ttl and self._ttl > 0:
            ttl = int(self._ttl)
            pipe.expire(self._k_meta(session_id), ttl)
            pipe.expire(self._k_msgs(session_id), ttl)
            pipe.expire(key_art, ttl)
        results = pipe.execute()
        # rpush returns new length as first result
        try:
            new_len = int(results[0])
        except Exception:
            new_len = int(self._r.llen(key_art))
        return max(0, new_len - 1)

    def message_count(self, session_id: str) -> int:
        return int(self._r.llen(self._k_msgs(session_id)))


# Placeholder for the session store, to be initialized on app startup.
session_store: SessionStore | RedisSessionStore = SessionStore(ttl_seconds=86400)  # Default fallback

def initialize_session_store():
    """Initializes the session store based on settings."""
    global session_store
    if settings.enable_redis_sessions:
        # Avoid printing full URL with credentials
        print("Connecting to Redis (URL hidden)...")
        print(f"Redis sessions enabled: {settings.enable_redis_sessions}")
        try:
            _client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            _client.ping()
            print("Redis connection successful.")
            session_store = RedisSessionStore(
                _client, ttl_seconds=settings.session_ttl_seconds, key_prefix=settings.redis_key_prefix
            )
        except Exception as e:
            print(f"FATAL: Could not connect to Redis: {e}")
            print("Please ensure Redis is running and accessible at the configured URL.")
            # Fallback to in-memory store to allow server to run for other routes.
            print("Falling back to in-memory session store.")
            session_store = SessionStore(ttl_seconds=settings.session_ttl_seconds)
    else:
        print("Using in-memory session store.")
        session_store = SessionStore(ttl_seconds=settings.session_ttl_seconds)
