"""Step5 command-line arguments."""

from argparse import ArgumentParser


def add_step5_arguments(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group(title="step5")
    group.add_argument(
        "--reuse-indexer-on-recompute",
        action="store_true",
        default=False,
        help=(
            "Cache selected block IDs to skip the entire frozen Indexer during activation recomputation "
            "(default: disabled). Uses extra memory."
        ),
    )
    return parser
