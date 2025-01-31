import os
from pathlib import Path

TESTS_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
RESOURCES_DIR = TESTS_DIR / "resources"