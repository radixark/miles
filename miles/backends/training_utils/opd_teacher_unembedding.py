"""Local replica of the disaggregated OPD teacher's unembedding.

The teacher mesh only ever sends hidden states (see `get_hidden_states` in
`logit_processors.py` and `send_teacher_hidden_states` in `megatron_utils/actor.py`).
Those hidden states are already post-final-norm: Megatron's `TransformerBlock.forward`
applies `final_layernorm` internally before returning from `self.decoder(...)`, and the
disaggregated teacher's `output_layer.forward` is intercepted (see
`install_teacher_hidden_states_passthrough`) after that point. So this module lets the
student mesh reconstruct full-vocab teacher log-probs locally with just a linear
projection + log-softmax, using a one-time (not per-step) read of a single tensor out
of the teacher's checkpoint.
"""

from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
import torch.nn.functional as F

_OUTPUT_LAYER_KEY = "output_layer.weight"
_TIED_EMBEDDING_KEY = "embedding.word_embeddings.weight"


def _resolve_teacher_checkpoint_dir(checkpoint_root: str, ckpt_step: int | None) -> Path:
    root = Path(checkpoint_root)
    if ckpt_step is not None:
        return root / f"iter_{ckpt_step:07d}"
    tracker = (root / "latest_checkpointed_iteration.txt").read_text().strip()
    if tracker == "release":
        return root / "release"
    return root / f"iter_{int(tracker):07d}"


def load_teacher_output_layer(
    checkpoint_root: str,
    ckpt_step: int | None,
    *,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read only the unembedding weight out of a Megatron torch_dist checkpoint,
    without loading the rest of the (possibly huge) model.

    Falls back to the tied embedding weight when the model ties input/output
    embeddings (the default unless --untie-embeddings-and-output-weights is set):
    such checkpoints have no separate output_layer.weight at all.
    """
    checkpoint_dir = _resolve_teacher_checkpoint_dir(checkpoint_root, ckpt_step)
    reader = dist_cp.FileSystemReader(str(checkpoint_dir))
    available_keys = reader.read_metadata().state_dict_metadata.keys()
    output_layer_key = _OUTPUT_LAYER_KEY if _OUTPUT_LAYER_KEY in available_keys else _TIED_EMBEDDING_KEY

    state_dict = {output_layer_key: torch.empty((vocab_size, hidden_size), dtype=dtype)}
    dist_cp.load(state_dict, storage_reader=reader)
    return state_dict[output_layer_key]


class TeacherUnembedding(torch.nn.Module):
    """Applies the teacher's unembedding to received (already post-final-norm) hidden states.

    Requires `--tensor-model-parallel-size=1` on the student (see the validation in
    `miles.utils.arguments`): this module holds the full, unsharded vocab weight, and
    reconciling that against vocab-parallel-sharded student logits isn't implemented.
    """

    def __init__(self, output_layer_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output_layer_weight", output_layer_weight, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`[R, H]` hidden states -> `[R, V]` teacher full-vocab log-probs (float32)."""
        hidden_states = hidden_states.to(torch.float32)
        logits = F.linear(hidden_states, self.output_layer_weight.to(torch.float32))
        # TODO: this can blow up the memory as it fails back to torch softmax
        return torch.log_softmax(logits, dim=-1)


def build_teacher_unembedding(args: Namespace, *, device: torch.device) -> TeacherUnembedding:
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    output_layer_weight = load_teacher_output_layer(
        args.opd_teacher_load,
        args.opd_teacher_ckpt_step,
        hidden_size=args.hidden_size,
        vocab_size=args.padded_vocab_size,
        dtype=dtype,
    )
    return TeacherUnembedding(output_layer_weight.to(device=device)).to(device=device)
