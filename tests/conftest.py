import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PDP_DIR = REPO_ROOT / "pdp"

# The PDP code uses top-level imports (``import utils``, ``from models...``), so it needs its own
# directory on sys.path; the repo root makes ``autocheckout`` and ``tools`` importable without install.
for path in (REPO_ROOT, PDP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
