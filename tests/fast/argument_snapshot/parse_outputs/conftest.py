from pathlib import Path

import pytest


@pytest.fixture
def result_files(tmp_path: Path) -> Path:
    files = {
        "custom.yaml": "lr: 0.000009\neps_clip: 0.31\n",
        "fsdp.yaml": "lr: 0.000009\nweight_decay: 0.2\n",
        "invalid-fsdp.yaml": "unknown_snapshot_option: 1\n",
        "eval.yaml": "eval:\n  defaults:\n    n_samples_per_eval_prompt: 3\n  datasets:\n    - name: tiny\n      path: $FIXTURES/data.jsonl\n",
        "empty-eval.yaml": "eval:\n  datasets: []\n",
        "checkpoint/latest_checkpointed_iteration.txt": "7\n",
        "ref/latest_checkpointed_iteration.txt": "3\n",
        "data.jsonl": '{"prompt": "1+1", "label": "2"}\n',
        "data2.jsonl": '{"prompt": "2+2", "label": "4"}\n',
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.replace("$FIXTURES", str(tmp_path)))
    return tmp_path
