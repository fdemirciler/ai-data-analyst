from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (parent of backend) is on sys.path for 'backend' package imports
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
