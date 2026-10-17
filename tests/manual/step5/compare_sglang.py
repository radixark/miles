"""Compare SGLang's input-token log-probs against the reference golden outputs.

    python3 compare_sglang.py --golden /scratch/step5_golden/step5-5layer.pt \
        --url http://127.0.0.1:30000 --out /scratch/step5_golden/step5-5layer-sglang.json
"""

import argparse
import json
import urllib.request

import torch


def _query(url: str, ids: list[int], topk: int) -> dict:
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": topk,
    }
    request = urllib.request.Request(
        f"{url}/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["meta_info"]


def _compare(golden: dict, meta: dict) -> dict:
    # input_token_logprobs[i] = (logprob, token_id, text); entry 0 has no logprob.
    sglang = torch.tensor([entry[0] for entry in meta["input_token_logprobs"][1:]], dtype=torch.float32)
    reference = golden["token_logprobs"].float()
    diff = (sglang - reference).abs()
    # input_top_logprobs[i] is the distribution that predicts token i, so row i+1 pairs with golden row i.
    sglang_top1 = torch.tensor([row[0][1] for row in meta["input_top_logprobs"][1:]])
    golden_top1 = golden["topk_ids"][:-1, 0]
    return {
        "tokens": int(diff.numel()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "p99_abs_diff": float(diff.quantile(0.99)),
        "top1_agreement": float((sglang_top1 == golden_top1).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    report = []
    for golden in torch.load(args.golden)["results"]:
        stats = _compare(golden, _query(args.url, golden["ids"], args.topk))
        report.append({"length": len(golden["ids"]), **stats})
        print(json.dumps(report[-1]))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
