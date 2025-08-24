# Ensure the backend package root is importable as top-level for tests
# This allows imports like `from app...` when running pytest from the repo root.
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
