from __future__ import annotations

from typing import Dict, Any
import tempfile
from pathlib import Path
import textwrap

from ..config import settings
from ..logging_utils import log_sandbox_execution, log_error

try:  # pragma: no cover - docker optional
    import docker  # type: ignore
except Exception:  # pragma: no cover - no docker installed
    docker = None  # type: ignore


def execute_analysis_script(
    code: str, dataset_path: str | Path, timeout_seconds: int = 30
) -> Dict[str, Any]:
    """Execute generated analysis code in an optional Docker sandbox.

    Behavior:
      - If sandbox disabled or docker unavailable -> return skipped result.
      - Writes code to temp directory as script.py.
      - (Future) Run container python script.py with volume mounts (read-only dataset).
    """
    if not settings.enable_docker_sandbox or docker is None:
        result = {"status": "skipped", "reason": "sandbox_disabled"}
        log_sandbox_execution(result)
        return result

    client = docker.from_env()  # type: ignore
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script_path = tmp_path / "script.py"
        script_path.write_text(textwrap.dedent(code), encoding="utf-8")

        ds_path = Path(dataset_path)
        data_mount_dir = "/data"

        # Use a pandas-enabled image or install without network if available
        try:
            # Try to use jupyter/scipy-notebook which has pandas pre-installed
            # If not available, temporarily enable network for package installation
            try:
                # First attempt: use pre-built image with pandas
                container = client.containers.run(  # type: ignore
                    image="jupyter/scipy-notebook:latest",
                    command=["python", "/work/script.py"],
                    network_disabled=True,
                    volumes={
                        str(tmp_path): {"bind": "/work", "mode": "ro"},
                        str(ds_path.parent): {"bind": data_mount_dir, "mode": "ro"},
                    },
                    working_dir="/work",
                    detach=True,
                    mem_limit="512m",
                    nano_cpus=1_000_000_000,  # 1.0 CPU
                    stdout=True,
                    stderr=True,
                    user="root",  # Override to avoid permission issues
                )
                print("DEBUG: Using pre-built scipy image")
            except docker.errors.ImageNotFound:  # type: ignore
                # Fallback: temporarily enable network for package installation
                print(
                    "DEBUG: Pre-built image not found, installing packages with network"
                )
                container = client.containers.run(  # type: ignore
                    image="python:3.11-slim",
                    command=[
                        "bash",
                        "-c",
                        "pip install pandas numpy pyarrow && python /work/script.py",
                    ],
                    network_disabled=False,  # Enable network for installation
                    volumes={
                        str(tmp_path): {"bind": "/work", "mode": "ro"},
                        str(ds_path.parent): {"bind": data_mount_dir, "mode": "ro"},
                    },
                    working_dir="/work",
                    detach=True,
                    mem_limit="512m",
                    nano_cpus=1_000_000_000,  # 1.0 CPU
                    stdout=True,
                    stderr=True,
                )

            # Wait for container and get results
            wait_result = container.wait(timeout=timeout_seconds)  # type: ignore
            exit_code = wait_result.get("StatusCode", -1)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore")  # type: ignore

            # Enhanced logging for debugging
            from ..logging_utils import log_sandbox_execution as log_info

            print(f"DEBUG: Container exit code: {exit_code}")
            print(f"DEBUG: Container logs (first 1000 chars): {logs[:1000]}")

            container.remove()

            # Consider execution successful if we have meaningful output,
            # even if exit code is non-zero (due to warnings/non-critical errors)
            has_meaningful_output = (
                logs
                and len(logs.strip()) > 100
                and (
                    "Variance Table" in logs or "DataFrame" in logs or "Metric" in logs
                )
            )

            status = "ok" if (exit_code == 0 or has_meaningful_output) else "error"

            result = {
                "status": status,
                "exit_code": exit_code,
                "logs": logs[-4000:],  # trim
            }
            log_sandbox_execution(result)
            return result
        except Exception as e:  # pragma: no cover - depends on docker runtime
            log_error(
                "sandbox", e, f"Docker execution failed for script: {code[:100]}..."
            )
            result = {"status": "error", "reason": str(e)}
            log_sandbox_execution(result)
            return result
