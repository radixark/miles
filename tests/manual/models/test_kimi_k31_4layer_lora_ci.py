"""Temporary partial-model CI for Kimi-K3.1 (layers 0-3, first 64 experts; the K3 4-layer-64expert cut).

Run by hand on an 8-GPU H200 host; no CI job collects tests/manual. Two prerequisites keep it out of CI: the
partial is not on the Hub (stage the MXFP4 export at <model_dir>/Kimi-K3.1-4layer-64experts), and SGLang must
carry Kimi-K3.1 DSA with a BF16 KV cache and indexer top-k capture under TP attention.
dapo-math with 4096-token responses runs most samples past the 2048-key DSA top-k; rollout and trainer
share BF16 experts and KV, and R3 plus indexer replay pin the discrete selections.
"""

import os

from scripts.run_kimi_k3 import ScriptArgs, _download_dataset, _prepare_bf16, _prepare_torch_dist, _train


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Kimi-K3.1-4layer-64experts",
        train_mode="lora",
        mode="normal",
        task="dapo-math",
        # the pruned model scores 0 on dapo-math, which zeroes every advantage; a fixed pseudo-random
        # reward keeps the weights moving so the weight sync carries real deltas
        reward_model="deterministic_random",
        hardware="H200",
        num_nodes=1,
        num_gpus_per_node=8,
        lora_rank=32,
        lora_alpha=64,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        rollout_max_response_len=4096,
        rollout_max_concurrency=16,
        max_tokens_per_gpu=8192,
        check_lora_weight_equal=True,
        skip_saving=True,
        rollout_bf16=True,
        # SGLang sizes its top-k capture host buffers by KV capacity, which the 4-layer model's tiny KV puts
        # near 200 GB per rank; cap it so eight ranks fit in host memory
        sglang_max_total_tokens=131072,
        # SGLang's FP8 indexer and the trainer's BF16 one share 99.4% of the top-k past 2048 keys, but in some
        # micro-batches up to 6% of tokens fall below the check's 80% per-token overlap; a broken capture fails ~36%
        extra_env_vars="MILES_TEST_R3_THRESHOLD=0.1",
        extra_args=(
            "--ci-test --ci-disable-logprobs-checker --no-enable-sample-ownership-checker "
            "--sglang-kv-cache-dtype bfloat16 --use-rollout-routing-replay --use-rollout-indexer-replay "
        ),
    )


def prepare(args: ScriptArgs):
    _download_dataset(args)
    _prepare_bf16(args)
    _prepare_torch_dist(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
