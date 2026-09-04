"""Put the example on sys.path so its module imports by bare name, the way a
rollout loads it (``--custom-agent-function-path``)."""

import sys

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
