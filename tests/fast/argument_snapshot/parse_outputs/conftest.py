import json
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
        "hf/config.json": json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 128,
                "intermediate_size": 512,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 1024,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-05,
                "rope_theta": 10000.0,
                "tie_word_embeddings": True,
            }
        ),
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.replace("$FIXTURES", str(tmp_path)))
    return tmp_path
