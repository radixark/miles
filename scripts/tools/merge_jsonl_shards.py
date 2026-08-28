"""Merge ordered JSONL shards into one atomically published training artifact."""

import hashlib
import json
from pathlib import Path

from tap import Tap


class Arguments(Tap):
    input_dir: Path
    output_dataset: Path
    pattern: str = "shard-*.jsonl"
    expected_shards: int | None = None

    def configure(self) -> None:
        self.add_argument("--input-dir", type=Path, help="Directory containing ordered JSONL shards.")
        self.add_argument("--output-dataset", type=Path, help="Merged JSONL destination.")
        self.add_argument("--pattern", type=str, default="shard-*.jsonl")
        self.add_argument("--expected-shards", type=int, default=None)


def merge_jsonl_shards(
    input_dir: Path,
    *,
    output_dataset: Path,
    pattern: str,
    expected_shards: int | None,
) -> dict[str, object]:
    shards = sorted(input_dir.glob(pattern))
    if not shards:
        raise ValueError(f"no shards matching {pattern!r} under {input_dir}")
    if expected_shards is not None and len(shards) != expected_shards:
        raise ValueError(f"expected {expected_shards} shards, found {len(shards)}")

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dataset.with_name(f"{output_dataset.name}.partial")
    digest = hashlib.sha256()
    rows_written = 0
    bytes_written = 0
    with partial.open("wb") as output_stream:
        for shard in shards:
            last_byte: bytes | None = None
            with shard.open("rb") as input_stream:
                while chunk := input_stream.read(8 * 1024 * 1024):
                    output_stream.write(chunk)
                    digest.update(chunk)
                    rows_written += chunk.count(b"\n")
                    bytes_written += len(chunk)
                    last_byte = chunk[-1:]
            if last_byte not in {None, b"\n"}:
                raise ValueError(f"non-empty shard does not end with a newline: {shard}")

    partial.replace(output_dataset)
    return {
        "input_dir": str(input_dir.resolve()),
        "output_dataset": str(output_dataset.resolve()),
        "shards": len(shards),
        "rows_written": rows_written,
        "bytes_written": bytes_written,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    args = Arguments().parse_args()
    if args.expected_shards is not None and args.expected_shards <= 0:
        raise ValueError("expected_shards must be positive when provided")
    summary = merge_jsonl_shards(
        args.input_dir,
        output_dataset=args.output_dataset,
        pattern=args.pattern,
        expected_shards=args.expected_shards,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
