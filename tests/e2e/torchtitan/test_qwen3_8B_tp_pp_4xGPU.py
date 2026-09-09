from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=900, suite="stage-c-4-gpu-h200", labels=["torchtitan"])

# Tensor and pipeline parallelism together, which is where the two dialects the
# backend has to speak both bite. Tensor parallelism shards the vocabulary, and
# miles' loss reduces the softmax over the tp group, so the logits have to reach
# it as the local shard -- gathering them shifts every log prob by -ln(tp).
# Pipeline parallelism infers its stage buffers once, so every microbatch of the
# run is padded to --titan-seq-len, and the pad carries consecutive positions
# rather than zeros (thousands of one-token documents break linear attention).
# The model is the untied dense one truncated to four layers: torchtitan cannot
# tie lm_head to the embedding across stages, so the tied 0.6B is refused under PP.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-8B",
    titan_model_name="qwen3",
    titan_model_flavor="8B",
    num_gpus=4,
    num_layers=4,
    tp_size=2,
    pp_size=2,
    seq_len=4096,
    max_response_len=2048,
    with_ref=True,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
