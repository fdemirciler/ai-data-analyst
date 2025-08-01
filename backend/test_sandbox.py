#!/usr/bin/env python3

from security.sandbox import SandboxEnvironment, ExecutionLimits

# Test the sandbox
sandbox = SandboxEnvironment(ExecutionLimits())
result = sandbox.execute_code('print("test")')

print("Success:", result.success)
print("Output:", result.output)
print("Error:", result.error)

# Test with dataframe context like the real scenario
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
context = {"df": df}

result2 = sandbox.execute_code('print(f"DataFrame shape: {df.shape}")', context)

print("\nWith DataFrame:")
print("Success:", result2.success)
print("Output:", result2.output)
print("Error:", result2.error)
