"""
Test script for the security layer
==================================

This script tests the code validator and sandbox environment
to ensure they work correctly.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.security import (
    CodeValidator,
    ValidationLevel,
    SecurePythonExecutor,
    ExecutionLimits,
    execute_python_code_safely,
)


def test_code_validator():
    """Test the code validator"""
    print("🔍 Testing Code Validator...")

    validator = CodeValidator(ValidationLevel.MODERATE)

    # Test safe code
    safe_code = """
import pandas as pd
import numpy as np

data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
result = data.mean()
print("Mean values:", result)
    """

    result = validator.validate_code(safe_code)
    print(
        f"✅ Safe code validation: {result.is_safe} (Risk: {result.risk_level.value})"
    )
    if result.violations:
        print(f"   Violations: {result.violations}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")

    # Test dangerous code
    dangerous_code = """
import os
import subprocess

os.system('rm -rf /')
subprocess.call(['cat', '/etc/passwd'])
    """

    result = validator.validate_code(dangerous_code)
    print(
        f"🚫 Dangerous code validation: {result.is_safe} (Risk: {result.risk_level.value})"
    )
    print(f"   Violations: {result.violations}")


def test_sandbox_execution():
    """Test the sandbox execution"""
    print("\n🏖️ Testing Sandbox Execution...")

    # Test safe data analysis code
    safe_code = """
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': ['x', 'y', 'z', 'x', 'y']
})

print("Data shape:", data.shape)
print("Data types:")
print(data.dtypes)

# Basic analysis
numeric_data = data.select_dtypes(include=[np.number])
print("\\nNumeric columns summary:")
print(numeric_data.describe())

# Value counts
print("\\nCategory counts:")
print(data['C'].value_counts())

result = "Analysis completed successfully"
    """

    result = execute_python_code_safely(
        safe_code, limits=ExecutionLimits(max_execution_time=30.0)
    )

    print(f"✅ Safe code execution: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    if result.output:
        print(f"   Output:\n{result.output}")
    if result.error:
        print(f"   Error: {result.error}")

    # Test code with potential issues
    print("\n⚠️ Testing problematic code...")
    problematic_code = """
import time

print("Starting long computation...")
for i in range(10):
    time.sleep(0.1)  # Small sleep to simulate work
    print(f"Step {i}")

print("Computation finished!")
    """

    result = execute_python_code_safely(
        problematic_code,
        limits=ExecutionLimits(max_execution_time=5.0),  # Short timeout
    )

    print(f"⏱️ Long-running code: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    if result.output:
        print(f"   Output (truncated):\n{result.output[:500]}...")
    if result.error:
        print(f"   Error: {result.error}")


def test_secure_executor():
    """Test the SecurePythonExecutor class"""
    print("\n🛡️ Testing SecurePythonExecutor...")

    executor = SecurePythonExecutor()

    # Test with context data
    context_data = {"numbers": [1, 2, 3, 4, 5], "message": "Hello from context!"}

    code_with_context = """
print("Message from context:", message)
print("Numbers:", numbers)

# Work with the numbers
total = sum(numbers)
average = total / len(numbers)

print(f"Total: {total}")
print(f"Average: {average}")

# Return some results
analysis_result = {
    'total': total,
    'average': average,
    'count': len(numbers)
}
    """

    result = executor.execute(code_with_context, context_data)

    print(f"✅ Context execution: {result.success}")
    if result.output:
        print(f"   Output:\n{result.output}")
    if result.globals_after:
        print(f"   Results available: {list(result.globals_after.keys())}")


if __name__ == "__main__":
    print("🧪 Security Layer Test Suite")
    print("=" * 50)

    try:
        test_code_validator()
        test_sandbox_execution()
        test_secure_executor()

        print("\n🎉 All tests completed!")
        print("✅ Security layer is working correctly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
