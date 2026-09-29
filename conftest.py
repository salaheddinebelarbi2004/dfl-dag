"""Pytest bootstrap: make the flat module layout importable.

The project deliberately uses flat, sibling imports (e.g. ``from client import
FederatedLearningClient``) rather than a package, so the directories that hold
those modules are prepended to ``sys.path`` here.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for _sub in ("core", "deploy", "scripts", "experiments"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
