"""Put the tau2-bench example's modules on sys.path for this package's tests."""

import sys

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
