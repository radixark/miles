import json
from pathlib import Path

from miles.utils.sft_dataset_utils import RowStats, iter_sft_dataset, summarize_sft_dataset


def test_jsonl_reader_accepts_heterogeneous_nested_tool_schemas(tmp_path: Path) -> None:
    rows = [
        {
            "messages": [{"role": "assistant", "content": "one"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "first",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
        {
            "messages": [{"role": "assistant", "content": "two"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "second",
                        "parameters": {
                            "type": "object",
                            "properties": {"count": {"type": "integer", "minimum": 0}},
                        },
                    },
                }
            ],
        },
    ]
    dataset = tmp_path / "heterogeneous.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    results = list(iter_sft_dataset(dataset))

    assert [result[0] for result in results] == [0, 1]
    assert [result[1] for result in results] == rows
    assert [result[2] for result in results] == [None, None]


def test_jsonl_reader_reports_invalid_rows_and_continues(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text('{"messages": []}\nnot-json\n{"messages": []}\n', encoding="utf-8")

    results = list(iter_sft_dataset(dataset))

    assert results[0] == (0, {"messages": []}, None)
    assert results[1][0:2] == (1, None)
    assert "line 2: invalid JSON" in (results[1][2] or "")
    assert results[2] == (2, {"messages": []}, None)


def test_summary_counts_rows_above_training_thresholds(tmp_path: Path) -> None:
    summary = summarize_sft_dataset(
        tmp_path / "dataset.jsonl",
        rows_seen=3,
        schema={"messages": {"list"}},
        stats=[
            RowStats(total_tokens=8192, loss_tokens=100, response_span_tokens=100),
            RowStats(total_tokens=65537, loss_tokens=200, response_span_tokens=200),
            RowStats(total_tokens=262145, loss_tokens=300, response_span_tokens=300),
        ],
        max_seq_len=2_000_000,
        error_count=0,
    )

    assert summary["rows_above_token_threshold"] == {
        "8192": 2,
        "16384": 2,
        "32768": 2,
        "65536": 2,
        "131072": 1,
        "262144": 1,
    }
