"""Archived: generator and system interfaces of the three-arm framework.

The evidence types they use stay active in ``src/common/evidence.py``.
"""

from .generator import CallableGenerator, GenerationResult, Generator
from .system import System

__all__ = ["CallableGenerator", "GenerationResult", "Generator", "System"]
