"""Full-weight Qwen with a fresh Clef head, sharded with FSDP2."""

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import Qwen3_5ForConditionalGeneration

from examples.clef.joint_schema_model import ClefModel, JointSchemaHead
from miles.backends.fsdp_utils.adaptations.precision import apply_fp32_master


class TrainableClefModel(ClefModel):
    def __init__(self, language_model: Any, head: JointSchemaHead) -> None:
        super().__init__(language_model, head)
        # FSDP2 shards parameters along dimension zero. Store the upstream scalar
        # scales as length-one tensors during training and restore scalars on export.
        for name in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
            setattr(head, name, torch.nn.Parameter(getattr(head, name).detach().reshape(1)))
        self.train_backbone = True

    def forward(self, batch: dict[str, Any]) -> list[list[torch.Tensor]]:
        # Vision is unused in this text pilot. Bypass it and the enormous vocabulary
        # projection; the decision head consumes final hidden states, not LM logits.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_backbone):
            outputs = self.language_model.model.language_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        lexical_weight = self.language_model.get_output_embeddings().weight
        if not self.train_backbone:
            lexical_weight = lexical_weight.detach()
        return self.head(
            outputs.last_hidden_state,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            lexical_weight,
        )


def build_model(model_dir: str, head_config: dict[str, int], device: torch.device, *, pretrained_head: bool = False) -> TrainableClefModel:
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if backbone.config.text_config.hidden_size != head_config["hidden_size"]:
        raise ValueError("backbone hidden size does not match head")
    backbone.config.use_cache = False
    backbone.model.visual.requires_grad_(False)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = backbone.to(device=device)
    head = JointSchemaHead(**head_config).to(device=device)
    # Tiny backbone learning rates require FP32 master weights and Adam moments;
    # updating BF16 weights directly can round away nearly every optimizer step.
    model = TrainableClefModel(backbone, head)
    if pretrained_head:
        load_trained_head(model.head, Path(model_dir) / "joint_head.safetensors")
    return apply_fp32_master(model)


def load_trained_head(head: torch.nn.Module, path: Path) -> None:
    state = load_file(str(path), device="cpu")
    expected = head.state_dict()
    for name in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
        if name in state and name in expected:
            state[name] = state[name].reshape(expected[name].shape)
    head.load_state_dict(state, strict=True)


def shard_model(model: TrainableClefModel, world_size: int) -> TrainableClefModel:
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.language_model.model.language_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    # Keep the head one sharding unit: its attention layers read projection weights
    # directly. Sharding those Linear children separately would bypass their hooks.
    fully_shard(model.head, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    return model


def read_head_config(path: Path) -> dict[str, int]:
    return json.loads(path.read_text())
