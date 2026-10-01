import argparse
from typing import Any

from miles.utils.test_utils.snapshot import dump_snapshot


def snapshot_parser(parser: argparse.ArgumentParser) -> dict[str, Any]:
    destinations = dict.fromkeys([action.dest for action in parser._actions] + list(parser._defaults))
    return {
        "parser": parser,
        "effective_defaults": {dest: parser.get_default(dest) for dest in destinations},
    }
