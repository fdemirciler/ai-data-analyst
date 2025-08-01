#!/usr/bin/env python3

from security.sandbox import SandboxEnvironment, ExecutionLimits

# Test the sandbox with debugging
sandbox = SandboxEnvironment(ExecutionLimits())

# First, let's see what builtins are actually created
print("Safe builtins keys:", list(sandbox.safe_builtins.keys()))
print("Print in safe_builtins:", "print" in sandbox.safe_builtins)
if "print" in sandbox.safe_builtins:
    print("Print function:", sandbox.safe_builtins["print"])

# Test simple code
result = sandbox.execute_code('print("test")')
print("\nExecution result:")
print("Success:", result.success)
print("Output:", result.output)
print("Error:", result.error)

# Test checking available builtins in execution
result2 = sandbox.execute_code(
    "print(list(__builtins__.keys()) if isinstance(__builtins__, dict) else dir(__builtins__))"
)
print("\nBuiltins check:")
print("Success:", result2.success)
print("Output:", result2.output)
print("Error:", result2.error)
