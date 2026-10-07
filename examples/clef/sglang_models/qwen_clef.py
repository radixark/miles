"""Execute Clef's full schema head as a SGLang prefill-only embedding model.

The local SystemOne gateway supplies exact span metadata through a content-addressed
sidecar. Full prefill, TP=1, and no prefix reuse are required: the head attends to
every input token and uses the complete vocabulary's output embeddings.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration as QwenBackbone

from examples.clef.joint_schema_model import EncodedQuestion, EncodedRecord, JointSchemaHead


def metadata_key(input_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(input_ids, separators=(",", ":")).encode()).hexdigest()


class ClefPooler(torch.nn.Module):
    def __init__(self, lexical_weight: torch.Tensor) -> None:
        super().__init__()
        model_path = Path(os.environ["CLEF_MODEL_PATH"])
        self.metadata_dir = Path(os.environ["CLEF_METADATA_DIR"])
        config = json.loads((model_path / "joint_head_config.json").read_text())
        self.head = JointSchemaHead(**config)
        self.head.load_state_dict(load_file(model_path / "joint_head.safetensors"), strict=True)
        self.head.to(device=lexical_weight.device, dtype=lexical_weight.dtype).eval()
        # Do not register a second copy of the backbone's lm_head parameter.
        object.__setattr__(self, "lexical_weight", lexical_weight)

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor, forward_batch: object) -> EmbeddingPoolerOutput:
        if not forward_batch.forward_mode.is_extend():
            raise ValueError("Clef requires a complete prefill, without decoding")
        lengths = forward_batch.extend_seq_lens_cpu
        ids = forward_batch.input_ids
        if sum(lengths) != hidden_states.shape[0]:
            raise ValueError("Clef received partial or padded hidden states")
        outputs = []
        offset = 0
        for length in lengths:
            token_ids = ids[offset : offset + length]
            values = token_ids.tolist()
            metadata_path = self.metadata_dir / (metadata_key(values) + ".json")
            data = json.loads(metadata_path.read_text())
            if data["input_ids"] != values:
                raise ValueError("Clef sidecar does not match the complete prefill")
            questions = tuple(
                EncodedQuestion(
                    question_id=q["question_id"], question_type=q["question_type"],
                    question_span=tuple(q["question_span"]),
                    option_spans=tuple(tuple(span) for span in q["option_spans"]),
                    option_ids=tuple(q["option_ids"]),
                )
                for q in data["questions"]
            )
            record = EncodedRecord(tuple(values), questions, data["record_id"])
            logits = self.head(
                hidden_states[offset : offset + length].unsqueeze(0),
                token_ids.unsqueeze(0), torch.ones_like(token_ids).unsqueeze(0),
                [record], self.lexical_weight,
            )[0]
            outputs.append(torch.cat([field.float().softmax(-1) for field in logits]))
            offset += length
        return EmbeddingPoolerOutput(embeddings=outputs)


class Qwen3_5ForConditionalGeneration(QwenBackbone):
    def __init__(self, config: object, quant_config: object = None, prefix: str = "") -> None:
        super().__init__(config, quant_config, prefix)
        if self.lm_head.weight.shape[0] < config.text_config.vocab_size:
            raise ValueError("Clef serving currently requires TP=1")
        self.pooler = ClefPooler(self.lm_head.weight)


EntryClass = Qwen3_5ForConditionalGeneration
