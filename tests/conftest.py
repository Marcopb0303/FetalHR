"""Put ``src/`` on the path so tests import ``analyze.*`` / ``constants`` the same way
``PYTHONPATH=src python src/analyze/main.py`` does.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
