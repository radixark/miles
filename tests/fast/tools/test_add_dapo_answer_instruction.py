import json

from tools.add_dapo_answer_instruction import ANSWER_INSTRUCTION, add_instruction


def test_the_instruction_ends_the_last_message_and_nothing_else_changes(tmp_path):
    rows = [
        {"prompt": [{"role": "user", "content": "Find $x$."}], "label": "73"},
        {"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "Find y."}], "label": "4"},
    ]
    src = tmp_path / "aime.jsonl"
    src.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dst = tmp_path / "out" / "aime-dapo.jsonl"

    add_instruction(src, dst)

    out = [json.loads(line) for line in dst.read_text().splitlines()]
    assert [row["prompt"][-1]["content"] for row in out] == [
        "Find $x$." + ANSWER_INSTRUCTION,
        "Find y." + ANSWER_INSTRUCTION,
    ]
    assert out[1]["prompt"][0] == rows[1]["prompt"][0]
    assert [row["label"] for row in out] == ["73", "4"]
    assert not list(dst.parent.glob("*.partial"))
