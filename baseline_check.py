import sys
from pathlib import Path

import app

print("Baseline check OK")
print(f"Python executable: {sys.executable}")
print(f"Project root: {Path.cwd()}")
print("App import OK")
