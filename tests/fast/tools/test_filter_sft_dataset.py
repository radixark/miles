import json
from pathlib import Path

import scripts.tools.filter_sft_dataset as filter_module
from miles.utils.sft_dataset_utils import RowStats


def test_filter_dataset_keeps_valid_rows_and_audits_expected_rejections(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {"id": "keep", "messages": [{"role": "assistant", "content": "ok"}]},
        {"id": "targetless", "messages": [{"role": "user", "content": "question"}]},
        {"id": "long", "messages": [{"role": "assistant", "content": "long"}]},
    ]
    dataset = tmp_path / "source.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output_dataset = tmp_path / "filtered.jsonl"
    reject_report = tmp_path / "rejected.jsonl"

    def fake_validate_row(row: dict[str, object], **_: object) -> RowStats:
        if row["id"] == "targetless":
            raise ValueError("messages must contain at least one non-empty assistant turn")
        if row["id"] == "long":
            raise ValueError("rendered length 262145 exceeds max_seq_len=262144")
        return RowStats(total_tokens=10, loss_tokens=4, response_span_tokens=4)

    monkeypatch.setattr(filter_module, "validate_sft_row", fake_validate_row)

    summary, unexpected_count = filter_module.filter_dataset(
        dataset,
        output_dataset=output_dataset,
        reject_report=reject_report,
        input_key="messages",
        tools_key="tools",
        max_seq_len=262144,
        mask_generator=object(),
    )

    assert json.loads(output_dataset.read_text(encoding="utf-8")) == rows[0]
    rejected = [json.loads(line) for line in reject_report.read_text(encoding="utf-8").splitlines()]
    assert [row["reason"] for row in rejected] == ["no_assistant_target", "over_max_seq_len"]
    assert summary["rows_seen"] == 3
    assert summary["validated_rows"] == 1
    assert summary["rejection_reasons"] == {
        "no_assistant_target": 1,
        "over_max_seq_len": 1,
    }
    assert unexpected_count == 0


def test_filter_dataset_reports_unexpected_invalid_rows(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "source.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")

    def fake_validate_row(*_: object, **__: object) -> RowStats:
        raise ValueError("messages must be a non-empty list")

    monkeypatch.setattr(filter_module, "validate_sft_row", fake_validate_row)

    summary, unexpected_count = filter_module.filter_dataset(
        dataset,
        output_dataset=tmp_path / "filtered.jsonl",
        reject_report=tmp_path / "rejected.jsonl",
        input_key="messages",
        tools_key="tools",
        max_seq_len=262144,
        mask_generator=object(),
    )

    assert summary["rejection_reasons"] == {"invalid_row": 1}
    assert unexpected_count == 1
