"""Convenience launcher for the FastAPI backend.

Usage (PowerShell):
  python run_server.py           # runs from project root or backend dir
  ENABLE_LLM=true python run_server.py

Resolves correct module path based on current working directory to avoid
'import backend.app.main' vs 'app.main' confusion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import uvicorn


def main():
    cwd = Path.cwd()
    # Detect if we're inside the backend directory (folder contains requirements.txt and app/)
    if (cwd / "app").is_dir() and (cwd / "requirements.txt").is_file():
        module_path = "app.main:app"
    else:
        # Assume project root -> backend/ exists
        if (cwd / "backend" / "app").is_dir():
            module_path = "backend.app.main:app"
        else:
            print(
                "Could not locate backend application package. Run from project root or backend directory."
            )
            sys.exit(1)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "true").lower() in {"1", "true", "yes"}

    print(f"Starting server: {module_path} on {host}:{port} reload={reload}")
    uvicorn.run(module_path, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
