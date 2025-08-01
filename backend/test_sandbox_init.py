#!/usr/bin/env python3

from security.sandbox import SandboxEnvironment, ExecutionLimits

# Debug the sandbox initialization step by step
print("Creating ExecutionLimits...")
limits = ExecutionLimits()
print("Limits created:", limits)

print("\nCreating SandboxEnvironment...")
sandbox = SandboxEnvironment(limits)

print("\nChecking sandbox attributes...")
print("Has safe_builtins:", hasattr(sandbox, "safe_builtins"))
print("Has safe_globals:", hasattr(sandbox, "safe_globals"))

if hasattr(sandbox, "safe_builtins"):
    print("safe_builtins type:", type(sandbox.safe_builtins))
    print(
        "safe_builtins keys:",
        list(sandbox.safe_builtins.keys()) if sandbox.safe_builtins else "Empty!",
    )

# Let's manually call the method to see what happens
print("\nManually calling _create_safe_builtins...")
try:
    manual_builtins = sandbox._create_safe_builtins()
    print("Manual builtins keys:", list(manual_builtins.keys()))
    print("Print in manual builtins:", "print" in manual_builtins)
except Exception as e:
    print("Error calling _create_safe_builtins:", e)

# Check if it's an issue with the validator
print("\nChecking validator...")
print("Has validator:", hasattr(sandbox, "validator"))
if hasattr(sandbox, "validator"):
    print("Validator type:", type(sandbox.validator))
