#!/usr/bin/env python3

# Test what's available in __builtins__
print("Type of __builtins__:", type(__builtins__))
print("Has print attribute:", hasattr(__builtins__, "print"))

if isinstance(__builtins__, dict):
    print("__builtins__ is dict with keys:", list(__builtins__.keys()))
    print("Print in dict:", "print" in __builtins__)
else:
    print("__builtins__ is module/object")
    print("Has print attr:", hasattr(__builtins__, "print"))

# Check if we can get print directly
try:
    print_func = getattr(__builtins__, "print")
    print("Got print function:", print_func)
except AttributeError as e:
    print("Could not get print:", e)

# Test the actual logic from sandbox
safe_builtin_names = {"print", "len", "str", "int"}
safe_builtins = {}

for name in safe_builtin_names:
    print(f"Checking {name}...")
    if hasattr(__builtins__, name):
        value = getattr(__builtins__, name)
        safe_builtins[name] = value
        print(f"  Added {name}: {value}")
    else:
        print(f"  {name} not found")

print("Final safe_builtins:", list(safe_builtins.keys()))
