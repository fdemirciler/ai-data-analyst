"""
Test script for Windows Resource Limits
======================================

This script tests the new cross-platform resource management system
to ensure that memory and CPU limits are properly enforced on Windows.
"""

import os
import sys
import time
import logging

# Add the backend to the path so we can import the security module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.security.sandbox import SandboxEnvironment, ExecutionLimits
from backend.security.resource_manager import create_resource_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_memory_limit():
    """Test memory limit enforcement"""
    logger.info("Testing memory limit enforcement...")

    # First, get the current memory usage to set a reasonable limit
    import psutil

    current_process = psutil.Process()
    current_memory_mb = current_process.memory_info().rss / 1024 / 1024

    # Set limit to current usage + 100MB for the test
    memory_limit = int(current_memory_mb) + 100
    logger.info(
        f"Current memory usage: {current_memory_mb:.1f}MB, setting limit to {memory_limit}MB"
    )

    # Create sandbox with reasonable memory limit
    limits = ExecutionLimits(
        max_memory_mb=memory_limit, max_execution_time=15.0, max_cpu_time=15.0
    )

    sandbox = SandboxEnvironment(limits)

    # Code that tries to allocate a lot of memory (200MB more than limit)
    memory_hog_code = f"""
import time
# Try to allocate {memory_limit + 200}MB of memory
big_list = []
chunk_size = 1000000  # 1MB per chunk
target_chunks = {memory_limit + 200}  # Target size in MB

print(f"Starting memory allocation test...")
print(f"Target allocation: {{target_chunks}}MB")

for i in range(target_chunks):
    big_list.append('x' * chunk_size)  # 1MB each
    if i % 10 == 0 and i > 0:
        print(f"Allocated {{i}}MB so far...")
        time.sleep(0.1)  # Give monitoring thread time to check
print(f"Memory allocation completed: {{len(big_list)}}MB")
"""

    result = sandbox.execute_code(memory_hog_code)

    logger.info(f"Memory test result: success={result.success}")
    logger.info(f"Output: {result.output}")
    if result.error:
        logger.info(f"Error: {result.error}")

    return result


def test_cpu_limit():
    """Test CPU time limit enforcement"""
    logger.info("Testing CPU time limit enforcement...")

    # Create sandbox with low CPU limit (2 seconds)
    limits = ExecutionLimits(
        max_memory_mb=256,
        max_execution_time=10.0,
        max_cpu_time=2.0,  # Only 2 seconds of CPU time
    )

    sandbox = SandboxEnvironment(limits)

    # Code that runs a CPU-intensive task for a long time
    cpu_hog_code = """
import time
start_time = time.time()
count = 0
# CPU-intensive loop that should exceed 2 seconds
while time.time() - start_time < 5:  # Try to run for 5 seconds
    for i in range(100000):
        count += i * i  # Some computation
    if count % 1000000000 == 0:
        print(f"CPU intensive work: count={count}, elapsed={time.time() - start_time:.1f}s")
print(f"CPU work completed: count={count}")
"""

    result = sandbox.execute_code(cpu_hog_code)

    logger.info(f"CPU test result: success={result.success}")
    logger.info(f"Output: {result.output}")
    if result.error:
        logger.info(f"Error: {result.error}")

    return result


def test_normal_execution():
    """Test that normal code still works"""
    logger.info("Testing normal code execution...")

    limits = ExecutionLimits(
        max_memory_mb=256, max_execution_time=10.0, max_cpu_time=5.0
    )

    sandbox = SandboxEnvironment(limits)

    # Normal code that should work fine
    normal_code = """
import pandas as pd
import numpy as np

# Create a small dataset
data = {
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
}
df = pd.DataFrame(data)
print("DataFrame created:")
print(df)

# Do some simple calculations
mean_x = df['x'].mean()
mean_y = df['y'].mean()
print(f"Mean x: {mean_x}, Mean y: {mean_y}")

result = mean_x + mean_y
print(f"Result: {result}")
"""

    result = sandbox.execute_code(normal_code)

    logger.info(f"Normal test result: success={result.success}")
    logger.info(f"Output: {result.output}")
    if result.error:
        logger.info(f"Error: {result.error}")

    return result


def test_resource_manager_info():
    """Test resource manager information"""
    logger.info("Testing resource manager creation...")

    manager = create_resource_manager()
    logger.info(f"Resource manager type: {type(manager).__name__}")

    # Test setting limits
    memory_result = manager.set_memory_limit(100)
    cpu_result = manager.set_cpu_limit(5.0)

    logger.info(f"Memory limit set: {memory_result}")
    logger.info(f"CPU limit set: {cpu_result}")

    # Test monitoring (briefly)
    manager.start_monitoring()
    time.sleep(1)
    usage = manager.get_current_usage()
    manager.stop_monitoring()

    if usage:
        logger.info(
            f"Current usage: {usage.memory_mb:.1f}MB, {usage.execution_time:.1f}s"
        )
    else:
        logger.info("No usage information available")

    manager.cleanup()


if __name__ == "__main__":
    logger.info("Starting Windows Resource Limits Test Suite")
    logger.info(f"Platform: {os.name}")
    logger.info("=" * 50)

    try:
        # Test resource manager info
        test_resource_manager_info()
        logger.info("=" * 50)

        # Test normal execution first
        normal_result = test_normal_execution()
        logger.info("=" * 50)

        # Test memory limits (may take some time)
        memory_result = test_memory_limit()
        logger.info("=" * 50)

        # Test CPU limits
        cpu_result = test_cpu_limit()
        logger.info("=" * 50)

        # Summary
        logger.info("TEST SUMMARY:")
        logger.info(f"Normal execution: {'PASS' if normal_result.success else 'FAIL'}")
        logger.info(
            f"Memory limit test: {'ENFORCED' if not memory_result.success else 'NOT ENFORCED'}"
        )
        logger.info(
            f"CPU limit test: {'ENFORCED' if not cpu_result.success else 'NOT ENFORCED'}"
        )

        if not memory_result.success or not cpu_result.success:
            logger.info("✅ Resource limits are being enforced!")
        else:
            logger.warning("⚠️  Resource limits may not be working properly")

    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
