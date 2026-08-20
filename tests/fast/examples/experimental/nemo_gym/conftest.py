"""Put the NeMo Gym example's modules on sys.path for this package's tests.

Same reason as the openenv sibling: a rollout loads them by path, so they are
importable only by bare name. One insert here replaces the copy the test file
carried next to its own imports.
"""

import sys

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
