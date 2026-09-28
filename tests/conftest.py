import os
import sys
from pathlib import Path

# The Mac test venv reuses the base interpreter's torch (--system-site-packages), and that interpreter
# also has TensorFlow, which transformers would import and which aborts the process. Must be set
# before transformers is imported. Harmless on the VM, where TensorFlow is not installed.
os.environ.setdefault("USE_TF", "0")
# Tests never download weights; without this, transformers/timm spend about a minute per model build
# waiting on the Hugging Face Hub when it is unreachable.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
PDP_DIR = REPO_ROOT / "pdp"

# The PDP code uses top-level imports (``import utils``, ``from models...``), so it needs its own
# directory on sys.path; the repo root makes ``autocheckout`` and ``tools`` importable without install.
for path in (REPO_ROOT, PDP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
