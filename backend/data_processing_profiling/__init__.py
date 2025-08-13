"""Data processing package providing cleaning, type inference, profiling and an orchestrated pipeline (relocated under backend)."""

from .pipeline import run_processing_pipeline  # noqa: F401

__all__ = ["run_processing_pipeline"]
