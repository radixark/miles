"""Policy models the τ²-bench recipe can train, and how each one runs.

``run.py`` holds the tau2 workload (domain, batch sizes, GRPO, eval) and takes
everything model-specific from one entry here, selected by ``--model-name``:
checkpoint, GPUs and parallelism, SGLang parsers, TITO family, chat-template
settings and sequence budgets. To train another model, add an entry.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PolicyModel:
    hf_repo: str
    megatron_model_type: str
    # The node count the entry was validated on; --num-nodes overrides it.
    num_nodes: int
    num_gpus_per_node: int
    # Training is data-parallel over what TP x CP leaves; each sample is one whole
    # conversation, split over the context-parallel ranks.
    tensor_parallel_size: int
    context_parallel_size: int
    # Keep the Adam state in host memory, as in scripts/run_qwen3_dense.py.
    optimizer_cpu_offload: bool
    rollout_num_gpus_per_engine: int
    sglang_mem_fraction_static: float
    # Both parsers are load-bearing: without the tool-call parser the policy's tool calls
    # stay plain text, and without the reasoning parser its thinking reaches tau2 as replies.
    sglang_reasoning_parser: str
    sglang_tool_call_parser: str
    tito_model: str
    chat_template_kwargs: dict[str, str]
    # Per policy turn, thinking included; max_seq_len caps the whole conversation.
    rollout_max_response_len: int
    max_seq_len: int


POLICY_MODELS: dict[str, PolicyModel] = {
    # Thinks before every reply, and its template keeps each turn's thinking in the
    # history, so the recipe asks for the template's low reasoning effort; higher
    # efforts need a longer max_seq_len. Validated on 2 x 8 H200.
    "Qwen3.8-27B": PolicyModel(
        hf_repo="Qwen/Qwen3.8-27B",
        megatron_model_type="qwen3.8-27B",
        num_nodes=2,
        num_gpus_per_node=8,
        tensor_parallel_size=4,
        context_parallel_size=2,
        optimizer_cpu_offload=True,
        rollout_num_gpus_per_engine=1,
        sglang_mem_fraction_static=0.8,
        sglang_reasoning_parser="qwen3",
        sglang_tool_call_parser="qwen3_coder",
        tito_model="qwen38small",
        chat_template_kwargs={"reasoning_effort": "low"},
        rollout_max_response_len=8192,
        max_seq_len=32768,
    ),
}

# The --model-name choices, read from the table so the two cannot drift.
PolicyModelName = Literal[tuple(POLICY_MODELS)]
