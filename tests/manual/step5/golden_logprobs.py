"""Record reference per-token log probabilities for fixed prompts.

Each prompt runs as one prefill; vocabulary shards are gathered before log-softmax.
See the README for checkpoint preparation and invocation.
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist


def _gather_vocab(local_logits: torch.Tensor, world: int, vocab: int) -> torch.Tensor:
    shards = [torch.empty_like(local_logits) for _ in range(world)]
    dist.all_gather(shards, local_logits.contiguous())
    return torch.cat(shards, dim=-1)[:, :vocab].float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-code", required=True)
    parser.add_argument("--tp-dir", required=True)
    parser.add_argument("--prompts", required=True, help="JSON list of token-id lists")
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--valid-vocab",
        type=int,
        default=None,
        help="normalize over the first N token ids, as SGLang does with valid_vocab_size (the tokenizer size)",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.ref_code)
    import transformers.modeling_utils as modeling_utils
    from generate import _load_rank_config, _load_rank_model, _resolve_device  # noqa: E402

    # The reference loads per-rank shard files by path. With peft installed, from_pretrained first
    # probes the path as a Hub repo for adapters and fails; the shards carry no adapters, so take
    # the same branch transformers takes when peft is absent.
    modeling_utils.maybe_load_adapters = lambda path, download_kwargs, **adapter_kwargs: (None, path, adapter_kwargs)

    device = _resolve_device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    config = _load_rank_config(args.tp_dir, rank)
    model = _load_rank_model(args.tp_dir, rank, config, device)
    vocab = args.valid_vocab or config.get_text_config().vocab_size

    results = []
    with torch.inference_mode():
        for ids in json.loads(open(args.prompts).read()):
            cache = model.allocate_cache(batch=1, max_tokens=len(ids), device=device)
            out = model(
                torch.tensor(ids, device=device, dtype=torch.long),
                cache,
                torch.tensor([len(ids)], device=device, dtype=torch.int32),
                return_all_logits=True,
            )
            logprobs = torch.log_softmax(_gather_vocab(out.logits, world, vocab), dim=-1)
            targets = torch.tensor(ids[1:], device=device, dtype=torch.long)
            top = logprobs.topk(args.topk, dim=-1)
            results.append(
                {
                    "ids": ids,
                    "token_logprobs": logprobs[:-1].gather(1, targets[:, None]).squeeze(1).cpu(),
                    "topk_ids": top.indices.cpu(),
                    "topk_logprobs": top.values.cpu(),
                }
            )
            del cache, out, logprobs

    if rank == 0:
        torch.save({"results": results, "tp": world, "vocab": vocab, "source": "stepfun inference/model.py"}, args.out)
        print(f"wrote {args.out}: {len(results)} prompts, lengths {[len(r['ids']) for r in results]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
