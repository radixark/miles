import hashlib
import json
from pathlib import Path

import pytest

from scripts.tools.merge_jsonl_shards import merge_jsonl_shards


def test_merge_jsonl_shards_preserves_lexical_shard_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "shards"
    input_dir.mkdir()
    (input_dir / "shard-02.jsonl").write_text('{"id":2}\n', encoding="utf-8")
    (input_dir / "shard-00.jsonl").write_text('{"id":0}\n', encoding="utf-8")
    (input_dir / "shard-01.jsonl").write_text('{"id":1}\n', encoding="utf-8")
    output_dataset = tmp_path / "merged.jsonl"

    summary = merge_jsonl_shards(
        input_dir,
        output_dataset=output_dataset,
        pattern="shard-*.jsonl",
        expected_shards=3,
    )

    content = output_dataset.read_bytes()
    assert [json.loads(line) for line in content.splitlines()] == [{"id": 0}, {"id": 1}, {"id": 2}]
    assert summary["shards"] == 3
    assert summary["rows_written"] == 3
    assert summary["bytes_written"] == len(content)
    assert summary["sha256"] == hashlib.sha256(content).hexdigest()


def test_merge_jsonl_shards_rejects_missing_shards(tmp_path: Path) -> None:
    (tmp_path / "shard-00.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 2 shards, found 1"):
        merge_jsonl_shards(
            tmp_path,
            output_dataset=tmp_path / "merged.jsonl",
            pattern="shard-*.jsonl",
            expected_shards=2,
        )
