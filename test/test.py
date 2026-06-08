import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = PROJECT_ROOT / "inference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from test.test_pulid import main


if __name__ == "__main__":
    main()

