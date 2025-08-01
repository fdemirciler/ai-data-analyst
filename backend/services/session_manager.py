"""
Redis-based session management for the Agentic Data Analysis Workflow.
Handles session creation, retrieval, persistence, and cleanup.
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio

import redis.asyncio as aioredis
from redis.asyncio import Redis

from ..config import get_settings
from ..models import (
    SessionData,
    ChatMessage,
    DataMetadata,
    SessionSummary,
    SessionStatus,
)
from ..utils import (
    RedisConnectionError,
    RedisOperationError,
    SessionNotFoundError,
    SessionExpiredError,
    get_logger,
    log_session_event,
)


class SessionManager:
    """Manages user sessions with Redis backend."""

    def __init__(self, redis_client: Optional[Redis] = None, settings=None):
        self.settings = settings or get_settings()
        self.redis_client = redis_client
        self.logger = get_logger(__name__)

        # Session configuration
        self.session_ttl = self.settings.session_ttl
        self.max_chat_history = self.settings.max_chat_history

        # Redis key prefixes
        self.session_prefix = "session:"
        self.session_list_key = "sessions:active"
        self.session_metadata_key = "sessions:metadata"

    async def _get_redis_client(self) -> Redis:
        """Get Redis client, creating connection if needed."""
        if self.redis_client is None:
            try:
                self.redis_client = aioredis.from_url(
                    self.settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                await self.redis_client.ping()
                self.logger.info("Redis connection established successfully")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                raise RedisConnectionError(self.settings.redis_url)

        return self.redis_client

    async def create_session(
        self, data_metadata: DataMetadata, session_id: Optional[str] = None
    ) -> SessionData:
        """
        Create a new session with uploaded data.

        Args:
            data_metadata: Metadata about the uploaded dataset
            session_id: Optional custom session ID

        Returns:
            SessionData: Created session data
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        redis_client = await self._get_redis_client()

        try:
            # Check if session already exists
            exists = await redis_client.exists(f"{self.session_prefix}{session_id}")
            if exists:
                raise RedisOperationError(
                    "create_session", f"Session {session_id} already exists"
                )

            # Create session data
            now = datetime.utcnow()
            session_data = SessionData(
                session_id=session_id,
                status=SessionStatus.ACTIVE,
                created_at=now,
                last_accessed=now,
                expires_at=now + timedelta(seconds=self.session_ttl),
                data_metadata=data_metadata,
                chat_history=[],
                active_workflows={},
                completed_workflows=[],
                total_queries=0,
                successful_queries=0,
                total_processing_time_ms=0,
                session_settings={},
            )

            # Store session in Redis
            await self._store_session(session_data)

            # Add to active sessions list
            await redis_client.sadd(self.session_list_key, session_id)

            # Store session metadata for quick access
            metadata = {
                "session_id": session_id,
                "filename": data_metadata.original_filename,
                "created_at": now.isoformat(),
                "status": SessionStatus.ACTIVE.value,
            }
            await redis_client.hset(
                self.session_metadata_key, session_id, json.dumps(metadata)
            )

            log_session_event(
                "session_created",
                session_id,
                filename=data_metadata.original_filename,
                file_size=data_metadata.file_size_bytes,
                rows=data_metadata.shape[0],
                columns=data_metadata.shape[1],
            )

            self.logger.info(f"Session created successfully: {session_id}")
            return session_data

        except Exception as e:
            self.logger.error(f"Failed to create session {session_id}: {e}")
            raise RedisOperationError("create_session", str(e))

    async def get_session(self, session_id: str) -> SessionData:
        """
        Retrieve session data by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionData: Session data

        Raises:
            SessionNotFoundError: If session doesn't exist
            SessionExpiredError: If session has expired
        """
        redis_client = await self._get_redis_client()

        try:
            # Get session data
            session_key = f"{self.session_prefix}{session_id}"
            session_json = await redis_client.get(session_key)

            if session_json is None:
                raise SessionNotFoundError(session_id)

            # Parse session data
            session_dict = json.loads(session_json)
            session_data = SessionData(**session_dict)

            # Check if session has expired
            if datetime.utcnow() > session_data.expires_at:
                await self._expire_session(session_id)
                raise SessionExpiredError(session_id)

            # Update last accessed time
            session_data.last_accessed = datetime.utcnow()
            await self._store_session(session_data)

            return session_data

        except (SessionNotFoundError, SessionExpiredError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {e}")
            raise RedisOperationError("get_session", str(e))

    async def update_session(self, session_data: SessionData) -> None:
        """
        Update session data in Redis.

        Args:
            session_data: Updated session data
        """
        redis_client = await self._get_redis_client()

        try:
            # Update last accessed time
            session_data.last_accessed = datetime.utcnow()

            # Store updated session
            await self._store_session(session_data)

            log_session_event(
                "session_updated",
                session_data.session_id,
                total_queries=session_data.total_queries,
                successful_queries=session_data.successful_queries,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to update session {session_data.session_id}: {e}"
            )
            raise RedisOperationError("update_session", str(e))

    async def add_chat_message(self, session_id: str, message: ChatMessage) -> None:
        """
        Add a chat message to session history.

        Args:
            session_id: Session identifier
            message: Chat message to add
        """
        try:
            session_data = await self.get_session(session_id)

            # Add message to history
            session_data.chat_history.append(message)

            # Trim history if it exceeds maximum
            if len(session_data.chat_history) > self.max_chat_history:
                session_data.chat_history = session_data.chat_history[
                    -self.max_chat_history :
                ]

            # Update statistics
            session_data.total_queries += 1
            if message.response_success:
                session_data.successful_queries += 1
            session_data.total_processing_time_ms += message.processing_time_ms

            # Update session
            await self.update_session(session_data)

            log_session_event(
                "message_added",
                session_id,
                message_id=message.message_id,
                success=message.response_success,
                processing_time_ms=message.processing_time_ms,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to add chat message to session {session_id}: {e}"
            )
            raise RedisOperationError("add_chat_message", str(e))

    async def get_chat_history(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> List[ChatMessage]:
        """
        Get chat history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List[ChatMessage]: Chat history messages
        """
        try:
            session_data = await self.get_session(session_id)

            # Apply pagination
            start_idx = max(0, len(session_data.chat_history) - offset - limit)
            end_idx = len(session_data.chat_history) - offset

            return session_data.chat_history[start_idx:end_idx]

        except Exception as e:
            self.logger.error(
                f"Failed to get chat history for session {session_id}: {e}"
            )
            raise RedisOperationError("get_chat_history", str(e))

    async def list_active_sessions(self) -> List[SessionSummary]:
        """
        List all active sessions.

        Returns:
            List[SessionSummary]: List of session summaries
        """
        redis_client = await self._get_redis_client()

        try:
            # Get active session IDs
            session_ids = await redis_client.smembers(self.session_list_key)

            summaries = []
            for session_id in session_ids:
                try:
                    # Get session metadata
                    metadata_json = await redis_client.hget(
                        self.session_metadata_key, session_id
                    )

                    if metadata_json:
                        metadata = json.loads(metadata_json)

                        # Get full session data for statistics
                        session_data = await self.get_session(session_id)

                        summary = SessionSummary(
                            session_id=session_id,
                            filename=session_data.data_metadata.original_filename,
                            status=session_data.status,
                            created_at=session_data.created_at,
                            last_accessed=session_data.last_accessed,
                            rows=session_data.data_metadata.shape[0],
                            columns=session_data.data_metadata.shape[1],
                            data_quality_score=session_data.data_metadata.data_quality_score,
                            total_queries=session_data.total_queries,
                            successful_queries=session_data.successful_queries,
                        )
                        summaries.append(summary)

                except (SessionNotFoundError, SessionExpiredError):
                    # Remove expired session from active list
                    await redis_client.srem(self.session_list_key, session_id)
                    await redis_client.hdel(self.session_metadata_key, session_id)
                except Exception as e:
                    self.logger.error(f"Error processing session {session_id}: {e}")

            return summaries

        except Exception as e:
            self.logger.error(f"Failed to list active sessions: {e}")
            raise RedisOperationError("list_active_sessions", str(e))

    async def delete_session(self, session_id: str) -> None:
        """
        Delete a session and all associated data.

        Args:
            session_id: Session identifier
        """
        redis_client = await self._get_redis_client()

        try:
            # Remove session data
            session_key = f"{self.session_prefix}{session_id}"
            await redis_client.delete(session_key)

            # Remove from active sessions list
            await redis_client.srem(self.session_list_key, session_id)

            # Remove metadata
            await redis_client.hdel(self.session_metadata_key, session_id)

            log_session_event("session_deleted", session_id)

            self.logger.info(f"Session deleted: {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            raise RedisOperationError("delete_session", str(e))

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            int: Number of sessions cleaned up
        """
        redis_client = await self._get_redis_client()

        try:
            session_ids = await redis_client.smembers(self.session_list_key)
            expired_count = 0

            for session_id in session_ids:
                try:
                    await self.get_session(session_id)
                except (SessionNotFoundError, SessionExpiredError):
                    await self._expire_session(session_id)
                    expired_count += 1
                except Exception as e:
                    self.logger.error(f"Error checking session {session_id}: {e}")

            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired sessions")

            return expired_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            raise RedisOperationError("cleanup_expired_sessions", str(e))

    async def extend_session(
        self, session_id: str, additional_seconds: int = None
    ) -> None:
        """
        Extend session expiration time.

        Args:
            session_id: Session identifier
            additional_seconds: Additional time in seconds (defaults to session_ttl)
        """
        if additional_seconds is None:
            additional_seconds = self.session_ttl

        try:
            session_data = await self.get_session(session_id)

            # Extend expiration time
            session_data.expires_at = datetime.utcnow() + timedelta(
                seconds=additional_seconds
            )

            await self.update_session(session_data)

            log_session_event(
                "session_extended",
                session_id,
                additional_seconds=additional_seconds,
                new_expiry=session_data.expires_at.isoformat(),
            )

        except Exception as e:
            self.logger.error(f"Failed to extend session {session_id}: {e}")
            raise RedisOperationError("extend_session", str(e))

    async def _store_session(self, session_data: SessionData) -> None:
        """Internal method to store session data in Redis."""
        redis_client = await self._get_redis_client()

        session_key = f"{self.session_prefix}{session_data.session_id}"
        session_json = session_data.model_dump_json()

        # Store with TTL
        await redis_client.setex(session_key, self.session_ttl, session_json)

    async def _expire_session(self, session_id: str) -> None:
        """Internal method to expire a session."""
        redis_client = await self._get_redis_client()

        # Remove from active sessions
        await redis_client.srem(self.session_list_key, session_id)
        await redis_client.hdel(self.session_metadata_key, session_id)

        log_session_event("session_expired", session_id)

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        redis_client = await self._get_redis_client()

        try:
            # Get total active sessions
            active_sessions = await redis_client.scard(self.session_list_key)

            # Get Redis memory info
            memory_info = await redis_client.info("memory")
            used_memory = memory_info.get("used_memory_human", "Unknown")

            return {
                "active_sessions": active_sessions,
                "redis_memory_usage": used_memory,
                "session_ttl_seconds": self.session_ttl,
                "max_chat_history": self.max_chat_history,
            }

        except Exception as e:
            self.logger.error(f"Failed to get session stats: {e}")
            raise RedisOperationError("get_session_stats", str(e))

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Redis connection closed")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def initialize_session_manager(
    redis_client: Optional[Redis] = None, settings=None
) -> SessionManager:
    """Initialize the global session manager with custom Redis client."""
    global _session_manager
    _session_manager = SessionManager(redis_client, settings)
    return _session_manager
