"""Entry point for the sharded federated learning system.

Run from the repository root:

    python main.py

Equivalent to running ``core/federated_learning_system.py`` directly, but
without needing to know the internal directory layout.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _sub in ("core", "deploy", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

if hasattr(sys.stdout, "reconfigure"):
    # Core modules print Unicode (e.g. check marks); the default Windows
    # console codec (cp1252) cannot encode them and raises UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from federated_learning_system import main  # noqa: E402

if __name__ == "__main__":
    main()
