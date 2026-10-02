"""Shared system interfaces."""

from .evidence import (
    Candidate,
    Evidence,
    ExperimentResult,
)
from .generator import (
    CallableGenerator,
    GenerationResult,
    Generator,
)
from .system import System

__all__ = [
    "Candidate",
    "Evidence",
    "ExperimentResult",
    "CallableGenerator",
    "GenerationResult",
    "Generator",
    "System",
]
