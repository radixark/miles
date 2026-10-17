"""Capture the BF16 input to the first full-attention layer for indexer parity checks.

Prompts include random token ids and tokenized text-file groups. See the README
for model flags, checkpoint loading, and output paths.
"""

import glob
import os

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_attr_wrapped_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model
from tokenizers import Tokenizer
from tools.convert_hf_to_torch_dist import add_conversion_args, get_args

import miles_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.logging_utils import configure_logger_raw
from miles_plugins.models.step5.layers import is_full_attention_layer

_RANDOM_LENGTHS = (5000, 32768)
_TEXT_LENGTH = 32768
_LOW, _HIGH = 1000, 120000  # clear of the special tokens at the top of the vocabulary


class _Captured(Exception):
    pass


def _add_capture_args(parser):
    add_conversion_args(parser)
    parser.add_argument("--text-files", nargs="*", default=[], help="glob patterns; one prompt per pattern")
    parser.add_argument("--out", required=True)
    return parser


def _prompts(args) -> dict[str, list[int]]:
    generator = torch.Generator().manual_seed(20260929)
    prompts = {f"random-{n}": torch.randint(_LOW, _HIGH, (n,), generator=generator).tolist() for n in _RANDOM_LENGTHS}
    tokenizer = Tokenizer.from_file(os.path.join(args.hf_checkpoint, "tokenizer.json"))
    for index, pattern in enumerate(args.text_files):
        ids = []
        for path in sorted(glob.glob(pattern, recursive=True)):
            with open(path, errors="replace") as f:
                ids += tokenizer.encode(f.read()).ids
            if len(ids) >= _TEXT_LENGTH:
                break
        assert len(ids) >= _TEXT_LENGTH, f"{pattern} tokenizes to only {len(ids)} tokens"
        prompts[f"text{index}-{_TEXT_LENGTH}"] = ids[:_TEXT_LENGTH]
    return prompts


def _load_weights(model, args) -> None:
    if args.load is not None:
        load_checkpoint(model, None, None, strict=True)
        return
    bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge.load_weights(model, args.hf_checkpoint, memory_efficient=True)


@torch.no_grad()
def _attention_input(model, layer, ids: list[int]) -> torch.Tensor:
    captured = []

    def capture(module, args, kwargs):
        captured.append((args[0] if args else kwargs["hidden_states"]).squeeze(1).cpu())
        raise _Captured

    handle = layer.self_attention.register_forward_pre_hook(capture, with_kwargs=True)
    cu_seqlens = torch.tensor([0, len(ids)], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=len(ids),
        max_seqlen_kv=len(ids),
        qkv_format="thd",
    )
    try:
        model(
            input_ids=torch.tensor([ids], device="cuda"),
            position_ids=None,
            attention_mask=None,
            packed_seq_params=packed,
        )
    except _Captured:
        pass
    finally:
        handle.remove()
    (hidden,) = captured
    return hidden


def main() -> None:
    configure_logger_raw()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    args = get_args(_add_capture_args)
    init(args)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    enforce_marked_param_dtypes(model)
    _load_weights(model, args)
    model[0].eval()
    decoder = get_attr_wrapped_model(model[0], "decoder")
    layer = next(layer for layer in decoder.layers if is_full_attention_layer(decoder.config, layer.layer_number))

    prompts = _prompts(args)
    hidden = {name: _attention_input(model[0], layer, ids) for name, ids in prompts.items()}
    if dist.get_rank() == 0:
        torch.save({"layer_index": layer.layer_number - 1, "ids": prompts, "hidden": hidden}, args.out)
        print(
            f"wrote {args.out}: layer {layer.layer_number - 1}, "
            + ", ".join(f"{k} {v.shape}" for k, v in hidden.items())
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
