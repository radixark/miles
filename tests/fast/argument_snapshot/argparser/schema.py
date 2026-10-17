import argparse
from typing import Any


def snapshot_parser(parser: argparse.ArgumentParser) -> dict[str, Any]:
    return {"parser": parser}
