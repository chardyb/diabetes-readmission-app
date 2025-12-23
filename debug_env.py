import sys
import os

print("--- Python Executable ---")
print(sys.executable)

print("\n--- System Path ---")
for p in sys.path:
    print(p)

print("\n--- Attempting Import ---")
try:
    import matplotlib
    print(f"SUCCESS: Matplotlib found at {matplotlib.__file__}")
except ImportError as e:
    print(f"FAILURE: {e}")

print("\n--- Checking Current Directory ---")
print(os.listdir('.'))
