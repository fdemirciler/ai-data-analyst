"""
Cross-Platform Resource Management
=================================

This module provides platform-specific resource limit enforcement
for secure code execution environments.
"""

import os
import platform
import threading
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    resource = None  # type: ignore
    HAS_RESOURCE = False

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Current resource usage statistics"""

    memory_mb: float
    cpu_percent: float
    execution_time: float
    pid: Optional[int] = None


class ResourceManager(ABC):
    """Abstract base class for platform-specific resource management"""

    @abstractmethod
    def set_memory_limit(self, limit_mb: int) -> bool:
        """Set memory limit in MB. Returns True if successfully set."""
        pass

    @abstractmethod
    def set_cpu_limit(self, limit_seconds: float) -> bool:
        """Set CPU time limit in seconds. Returns True if successfully set."""
        pass

    @abstractmethod
    def start_monitoring(self, pid: Optional[int] = None) -> None:
        """Start monitoring the process for resource usage."""
        pass

    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop monitoring the process."""
        pass

    @abstractmethod
    def get_current_usage(self) -> Optional[ResourceUsage]:
        """Get current resource usage."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the manager."""
        pass


class UnixResourceManager(ResourceManager):
    """Unix-specific resource manager using the resource module"""

    def __init__(self):
        self.memory_limit_mb: Optional[int] = None
        self.cpu_limit_seconds: Optional[float] = None
        self.monitoring_process: Optional[psutil.Process] = None
        self.start_time: Optional[float] = None

    def set_memory_limit(self, limit_mb: int) -> bool:
        """Set memory limit using resource.setrlimit"""
        if not HAS_RESOURCE:
            logger.warning("resource module not available")
            return False

        try:
            memory_limit = limit_mb * 1024 * 1024  # Convert MB to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))  # type: ignore
            self.memory_limit_mb = limit_mb
            logger.info(f"Memory limit set to {limit_mb}MB using resource.setrlimit")
            return True
        except (ValueError, OSError) as e:
            logger.error(f"Failed to set memory limit: {e}")
            return False

    def set_cpu_limit(self, limit_seconds: float) -> bool:
        """Set CPU time limit using resource.setrlimit"""
        if not HAS_RESOURCE:
            logger.warning("resource module not available")
            return False

        try:
            cpu_limit = int(limit_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))  # type: ignore
            self.cpu_limit_seconds = limit_seconds
            logger.info(f"CPU limit set to {limit_seconds}s using resource.setrlimit")
            return True
        except (ValueError, OSError) as e:
            logger.error(f"Failed to set CPU limit: {e}")
            return False

    def start_monitoring(self, pid: Optional[int] = None) -> None:
        """Start monitoring using psutil if available"""
        if HAS_PSUTIL:
            try:
                if pid is None:
                    pid = os.getpid()
                self.monitoring_process = psutil.Process(pid)
                self.start_time = time.time()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not start monitoring process {pid}: {e}")

    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.monitoring_process = None
        self.start_time = None

    def get_current_usage(self) -> Optional[ResourceUsage]:
        """Get current resource usage using psutil"""
        if not self.monitoring_process or not self.start_time:
            return None

        try:
            memory_info = self.monitoring_process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.monitoring_process.cpu_percent()
            execution_time = time.time() - self.start_time

            return ResourceUsage(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                execution_time=execution_time,
                pid=self.monitoring_process.pid,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def cleanup(self) -> None:
        """Clean up resources"""
        self.stop_monitoring()


class WindowsResourceManager(ResourceManager):
    """Windows-specific resource manager using psutil and threading"""

    def __init__(self):
        self.memory_limit_mb: Optional[int] = None
        self.cpu_limit_seconds: Optional[float] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.monitoring_process: Optional[psutil.Process] = None
        self.start_time: Optional[float] = None
        self.violation_callback: Optional[callable] = None

    def set_memory_limit(self, limit_mb: int) -> bool:
        """Set memory limit for monitoring"""
        if not HAS_PSUTIL:
            logger.warning("psutil not available for Windows resource management")
            return False

        self.memory_limit_mb = limit_mb
        logger.info(f"Memory limit set to {limit_mb}MB (Windows monitoring mode)")
        return True

    def set_cpu_limit(self, limit_seconds: float) -> bool:
        """Set CPU time limit for monitoring"""
        if not HAS_PSUTIL:
            logger.warning("psutil not available for Windows resource management")
            return False

        self.cpu_limit_seconds = limit_seconds
        logger.info(f"CPU limit set to {limit_seconds}s (Windows monitoring mode)")
        return True

    def set_violation_callback(self, callback: callable) -> None:
        """Set callback function to call when resource limits are violated"""
        self.violation_callback = callback

    def start_monitoring(self, pid: Optional[int] = None) -> None:
        """Start monitoring process resource usage"""
        if not HAS_PSUTIL:
            logger.warning("psutil not available for Windows resource monitoring")
            return

        try:
            if pid is None:
                pid = os.getpid()
            self.monitoring_process = psutil.Process(pid)
            self.start_time = time.time()
            self.should_stop.clear()

            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitor_process, daemon=True, name="ResourceMonitor"
            )
            self.monitor_thread.start()
            logger.info(f"Started resource monitoring for process {pid}")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Could not start monitoring process {pid}: {e}")

    def stop_monitoring(self) -> None:
        """Stop monitoring process"""
        self.should_stop.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.monitor_thread = None
        self.monitoring_process = None
        self.start_time = None

    def get_current_usage(self) -> Optional[ResourceUsage]:
        """Get current resource usage"""
        if not self.monitoring_process or not self.start_time:
            return None

        try:
            memory_info = self.monitoring_process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.monitoring_process.cpu_percent()
            execution_time = time.time() - self.start_time

            return ResourceUsage(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                execution_time=execution_time,
                pid=self.monitoring_process.pid,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def _monitor_process(self) -> None:
        """Monitor process and enforce resource limits"""
        if not self.monitoring_process or not self.start_time:
            return

        logger.debug("Starting resource monitoring loop")

        try:
            while not self.should_stop.is_set():
                try:
                    # Check if process still exists
                    if not self.monitoring_process.is_running():
                        logger.debug("Process no longer running, stopping monitor")
                        break

                    current_usage = self.get_current_usage()
                    if not current_usage:
                        break

                    # Memory limit check
                    if (
                        self.memory_limit_mb
                        and current_usage.memory_mb > self.memory_limit_mb
                    ):
                        logger.warning(
                            f"Memory limit exceeded: {current_usage.memory_mb:.1f}MB > "
                            f"{self.memory_limit_mb}MB"
                        )
                        self._handle_violation("memory", current_usage)
                        break

                    # CPU time limit check
                    if (
                        self.cpu_limit_seconds
                        and current_usage.execution_time > self.cpu_limit_seconds
                    ):
                        logger.warning(
                            f"CPU time limit exceeded: {current_usage.execution_time:.1f}s > "
                            f"{self.cpu_limit_seconds}s"
                        )
                        self._handle_violation("cpu_time", current_usage)
                        break

                    # Log resource usage periodically (every 5 seconds)
                    if int(current_usage.execution_time) % 5 == 0:
                        logger.debug(
                            f"Resource usage: {current_usage.memory_mb:.1f}MB, "
                            f"{current_usage.execution_time:.1f}s"
                        )

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.debug("Process no longer accessible, stopping monitor")
                    break

                # Check every 100ms
                self.should_stop.wait(0.1)

        except Exception as e:
            logger.error(f"Error in resource monitoring thread: {e}")

        logger.debug("Resource monitoring loop ended")

    def _handle_violation(self, violation_type: str, usage: ResourceUsage) -> None:
        """Handle resource limit violation"""
        if self.violation_callback:
            try:
                self.violation_callback(violation_type, usage)
            except Exception as e:
                logger.error(f"Error in violation callback: {e}")

        # Try to terminate the process
        if self.monitoring_process:
            try:
                logger.warning(
                    f"Terminating process {self.monitoring_process.pid} due to {violation_type} violation"
                )
                self.monitoring_process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.error(f"Could not terminate process: {e}")

    def cleanup(self) -> None:
        """Clean up resources"""
        self.stop_monitoring()


class NullResourceManager(ResourceManager):
    """Null resource manager for platforms without support"""

    def __init__(self):
        logger.warning(
            "Using null resource manager - no resource limits will be enforced"
        )

    def set_memory_limit(self, limit_mb: int) -> bool:
        logger.warning(
            f"Cannot set memory limit to {limit_mb}MB - no resource management available"
        )
        return False

    def set_cpu_limit(self, limit_seconds: float) -> bool:
        logger.warning(
            f"Cannot set CPU limit to {limit_seconds}s - no resource management available"
        )
        return False

    def start_monitoring(self, pid: Optional[int] = None) -> None:
        pass

    def stop_monitoring(self) -> None:
        pass

    def get_current_usage(self) -> Optional[ResourceUsage]:
        return None

    def cleanup(self) -> None:
        pass


def create_resource_manager() -> ResourceManager:
    """Factory function to create appropriate resource manager for current platform"""
    system = platform.system().lower()

    if system == "windows":
        if HAS_PSUTIL:
            logger.info("Creating Windows resource manager with psutil monitoring")
            return WindowsResourceManager()
        else:
            logger.warning("psutil not available, using null resource manager")
            return NullResourceManager()

    elif system in ("linux", "darwin", "unix"):
        if HAS_RESOURCE:
            logger.info("Creating Unix resource manager with resource.setrlimit")
            return UnixResourceManager()
        elif HAS_PSUTIL:
            logger.warning(
                "resource module not available, falling back to psutil monitoring"
            )
            return WindowsResourceManager()  # Works on Unix too
        else:
            logger.warning(
                "No resource management available, using null resource manager"
            )
            return NullResourceManager()

    else:
        logger.warning(f"Unsupported platform: {system}, using null resource manager")
        return NullResourceManager()
