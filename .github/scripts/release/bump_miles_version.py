"""Bump the miles version in setup.py.

Usage: python .github/scripts/release/bump_miles_version.py 0.3.0
"""

import re
import sys
from pathlib import Path

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:rc|\.post)\d+)?$")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    new_version = sys.argv[1]
    if not VERSION_RE.fullmatch(new_version):
        print(f"Invalid version {new_version!r}; expected X.Y.Z, X.Y.ZrcN, or X.Y.Z.postN", file=sys.stderr)
        return 1

    setup_py = Path(__file__).resolve().parents[3] / "setup.py"
    content = setup_py.read_text()
    new_content, n = re.subn(r'version="[^"]+"', f'version="{new_version}"', content)
    assert n == 1, f"expected exactly one version= assignment in setup.py, found {n}"
    setup_py.write_text(new_content)
    print(f"setup.py: version -> {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
