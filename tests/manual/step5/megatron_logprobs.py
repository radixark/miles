"""Compare Megatron Step5 log probabilities with reference goldens.

Load HF weights through mbridge or a converted torch_dist checkpoint via `--load`.
Each golden prompt runs as one THD prefill; see the README for invocation.
"""

import json
import os

import torch
import torch.distributed as dist
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model
from tools.convert_hf_to_torch_dist import add_conversion_args, get_args

import miles_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.logging_utils import configure_logger_raw


def _add_parity_args(parser):
    add_conversion_args(parser)
    parser.add_argument("--golden", required=True, help="output of golden_logprobs.py")
    parser.add_argument("--report", required=True)
    return parser


def _init_distributed() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))


def _load_weights(model, args) -> None:
    """From a torch_dist checkpoint with ``--load`` (the training path), else from the HF checkpoint."""
    if args.load is not None:
        load_checkpoint(model, None, None, strict=True)
        return
    bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge.load_weights(model, args.hf_checkpoint, memory_efficient=True)


@torch.no_grad()
def _logprobs(model, ids: list[int], vocab: int) -> torch.Tensor:
    length = len(ids)
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    logits = model(
        input_ids=torch.tensor([ids], device="cuda"),
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=length,
            max_seqlen_kv=length,
            qkv_format="thd",
        ),
    )
    logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
    return torch.log_softmax(logits[0, :, :vocab].float(), dim=-1)


def _compare(golden: dict, logprobs: torch.Tensor) -> dict:
    targets = torch.tensor(golden["ids"][1:], device=logprobs.device)
    token_logprobs = logprobs[:-1].gather(1, targets[:, None]).squeeze(1).cpu()
    diff = (token_logprobs - golden["token_logprobs"].float()).abs()
    return {
        "length": len(golden["ids"]),
        "mean_abs_diff": float(diff.mean()),
        "p99_abs_diff": float(diff.quantile(0.99)),
        "max_abs_diff": float(diff.max()),
        "top1_agreement": float((logprobs.argmax(-1).cpu() == golden["topk_ids"][:, 0]).float().mean()),
    }


def main() -> None:
    configure_logger_raw()
    _init_distributed()
    args = get_args(_add_parity_args)
    init(args)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    enforce_marked_param_dtypes(model)
    _load_weights(model, args)
    model[0].eval()

    golden = torch.load(args.golden)
    report = []
    for prompt in golden["results"]:
        report.append(_compare(prompt, _logprobs(model[0], prompt["ids"], golden["vocab"])))
        if dist.get_rank() == 0:
            print(json.dumps(report[-1]), flush=True)
    if dist.get_rank() == 0:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
