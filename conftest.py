"""Put the repo root on sys.path so `import src...` and `import api...` work.

pytest is meant to be run from the repo root (`pytest -v`). This shim just
makes the imports resilient if it's launched from somewhere else.

Also quiets TensorFlow's C++ logging before the first import, otherwise every
test session opens with a wall of oneDNN/cpu_feature_guard noise that buries
actual failures.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
