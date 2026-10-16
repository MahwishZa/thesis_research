"""Tests of the repository, in folders by purpose: spec, unit, integration, experiments, repository.

The tests import the packages under src/ without installing them: the folder goes on the search path first.
ROOT and SRC are the repository folder and its source folder, for every test that needs a path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
