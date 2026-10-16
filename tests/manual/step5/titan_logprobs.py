"""Compare the torchtitan Step5 model's log probabilities with reference goldens, on one GPU.

Loads the HF checkpoint through the package's state-dict adapter (FP8 experts dequantized on read) and casts
the weights as FSDP's mixed precision does. Each golden prompt runs alone, then all of them as one pack.

    python tests/manual/step5/titan_logprobs.py --hf-checkpoint /scratch/step5_slices/step5-5layer \
        --golden /scratch/step5_golden/step5-5layer.pt --report /scratch/step5_golden/step5-5layer-titan.json
"""

import argparse
import json

import torch
import torch.distributed.checkpoint as dcp

from miles.backends.torchtitan_utils.models.step5 import build_model_config
from miles.backends.torchtitan_utils.models.step5.layers import Step5RMSNorm
from miles.backends.torchtitan_utils.models.step5.state_dict_adapter import Step5StateDictAdapter
from miles.utils.hf_utils.config import load_hf_config


def _load(hf_checkpoint: str):
    config = build_model_config(load_hf_config(hf_checkpoint))
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cuda")
    model.init_states(buffer_device=torch.device("cuda"))
    adapter = Step5StateDictAdapter(config, hf_checkpoint)
    hf_state = adapter.to_hf(model.state_dict())
    dcp.load(hf_state, storage_reader=adapter.get_hf_storage_reader(hf_checkpoint), no_dist=True)
    model.load_state_dict(adapter.from_hf(hf_state))
    for module in model.modules():
        if not isinstance(module, Step5RMSNorm):
            for name, param in module.named_parameters(recurse=False):
                setattr(module, name, torch.nn.Parameter(param.detach().bfloat16(), requires_grad=False))
    return model.eval()


@torch.no_grad()
def _logprobs(model, sequences: list[list[int]], vocab: int) -> list[torch.Tensor]:
    tokens = torch.tensor([[token for ids in sequences for token in ids]], device="cuda")
    positions = torch.cat([torch.arange(len(ids), device="cuda") for ids in sequences]).unsqueeze(0)
    logprobs = torch.log_softmax(model(tokens, positions)[0, :, :vocab].float(), dim=-1)
    return list(logprobs.split([len(ids) for ids in sequences]))


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--golden", required=True, help="output of golden_logprobs.py")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    model = _load(args.hf_checkpoint)
    golden = torch.load(args.golden)
    prompts = golden["results"]
    alone = [_logprobs(model, [prompt["ids"]], golden["vocab"])[0] for prompt in prompts]
    packed = _logprobs(model, [prompt["ids"] for prompt in prompts], golden["vocab"])
    report = []
    for prompt, single, in_pack in zip(prompts, alone, packed, strict=True):
        report.append({**_compare(prompt, single), "packed_max_abs_diff": float((in_pack - single).abs().max())})
        print(json.dumps(report[-1]), flush=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
