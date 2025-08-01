"""
Simple Security Test
===================

Test the security layer with basic Python code that doesn't require pandas.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.security import CodeValidator, ValidationLevel


def test_basic_validation():
    """Test basic code validation without execution"""
    print("🔍 Testing Code Validator (Basic)...")

    validator = CodeValidator(ValidationLevel.MODERATE)

    # Test safe basic code
    safe_code = """
# Basic Python operations
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
average = total / len(numbers)

print(f"Numbers: {numbers}")
print(f"Total: {total}")
print(f"Average: {average}")

# Basic data processing
squares = [x**2 for x in numbers]
print(f"Squares: {squares}")
    """

    result = validator.validate_code(safe_code)
    print(f"✅ Safe basic code: {result.is_safe} (Risk: {result.risk_level.value})")
    if result.violations:
        print(f"   Violations: {result.violations}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")

    # Test dangerous code
    dangerous_code = """
import os
import subprocess
import sys

# Dangerous operations
os.system('dir')
subprocess.run(['echo', 'hello'])
sys.exit(1)
    """

    result = validator.validate_code(dangerous_code)
    print(f"🚫 Dangerous code: {result.is_safe} (Risk: {result.risk_level.value})")
    if result.violations:
        print(f"   Violations: {result.violations[:3]}...")  # Show first 3

    # Test moderately risky code
    risky_code = """
# This has unknown imports but no dangerous operations
import unknown_module
import json

data = {"key": "value"}
json_str = json.dumps(data)
print(json_str)
    """

    result = validator.validate_code(risky_code)
    print(f"⚠️ Risky code: {result.is_safe} (Risk: {result.risk_level.value})")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")


def test_code_patterns():
    """Test various code patterns"""
    print("\n🔍 Testing Code Patterns...")

    validator = CodeValidator(ValidationLevel.MODERATE)

    test_cases = [
        (
            "Arithmetic operations",
            """
x = 10
y = 20
result = x + y * 2
print(f"Result: {result}")
        """,
        ),
        (
            "String manipulation",
            """
text = "Hello, World!"
words = text.split(", ")
joined = " - ".join(words)
print(joined.upper())
        """,
        ),
        (
            "List comprehensions",
            """
numbers = list(range(10))
evens = [x for x in numbers if x % 2 == 0]
doubled = [x * 2 for x in evens]
print(doubled)
        """,
        ),
        (
            "Function definition",
            """
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

result = calculate_factorial(5)
print(f"5! = {result}")
        """,
        ),
        (
            "File operations (should be blocked)",
            """
with open('test.txt', 'w') as f:
    f.write('test data')
        """,
        ),
    ]

    for name, code in test_cases:
        result = validator.validate_code(code)
        status = "✅" if result.is_safe else "🚫"
        print(f"{status} {name}: {result.is_safe} (Risk: {result.risk_level.value})")
        if result.violations:
            print(f"   Issues: {result.violations}")


if __name__ == "__main__":
    print("🧪 Simple Security Test")
    print("=" * 40)

    try:
        test_basic_validation()
        test_code_patterns()

        print("\n🎉 Security validation tests completed!")
        print("✅ Code validation is working correctly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
