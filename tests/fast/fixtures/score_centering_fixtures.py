"""Small deterministic samples and SGLang reply fields for score-centering and sampling-mask tests."""

from argparse import Namespace

import numpy as np
import pybase64

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.rollout.generate_utils.rollout_topk_logprobs import append_rollout_topk_logprobs
from miles.utils.types import Sample


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return str(tokens)


def _args(**overrides: object) -> Namespace:
    values = dict(
        ci_test=False,
        enable_sample_ownership_checker=False,
        loss_type="score_centering",
        rollout_top_logprobs_num=3,
        rollout_sampling_logprobs_mode="selected",
        use_opd=False,
        use_tis=False,
        custom_tis_function_path=None,
        use_opsm=False,
        recompute_logprobs_via_prefill=False,
        sglang_speculative_algorithm=None,
        custom_pg_loss_reducer_function_path=None,
        multi_lora=False,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        custom_convert_samples_to_train_data_path=None,
        use_output_store=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        rollout_temperature=0.7,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        use_sampling_support_replay=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        length_penalty_coeff=0.0,
        length_penalty_min_tokens=1,
        use_dynamic_global_batch_size=False,
        balance_data=False,
        qkv_format="thd",
        true_on_policy_mode=False,
        allgather_cp=False,
        log_probs_chunk_size=2,
        vocab_size=8,
        calculate_per_token_loss=False,
        global_batch_size=1,
        entropy_coef=0,
        observe_training_entropy=False,
        use_kl_loss=False,
    )
    values.update(overrides)
    backend = MegatronArgsNamespace(
        **{name: values.pop(name) for name in ("vocab_size", "calculate_per_token_loss", "global_batch_size")}
    )
    sglang = SglangConfig(models=[], base_args={"speculative_algorithm": values["sglang_speculative_algorithm"]})
    return Namespace(**values, train_backend="megatron", backend=backend, sglang=sglang)


def _turn(prompt: list[int], output: list[int], probabilities: list[float]) -> Sample:
    logps = np.log(probabilities).tolist()
    sample = Sample(
        tokens=prompt + output,
        response_length=len(output),
        response=str(output),
        rollout_log_probs=logps,
        loss_mask=[1] * len(output),
        status=Sample.Status.COMPLETED,
        index=0,
        group_index=0,
        reward=1.0,
    )
    entries = [(logp, token, None) for logp, token in zip(logps, output, strict=True)]
    rows = [[*entries, *[(-np.inf, -1, None)] * (3 - len(entries))]] * len(output)
    append_rollout_topk_logprobs(sample, {"output_token_logprobs": entries, **_flat_output_top_logprobs(rows)}, 3)
    return sample


def _meta(output: list[int], probabilities: list[float]) -> dict:
    """An SGLang reply's meta_info with flat base64 candidates, padded to the 3 columns SGLang returns for
    ``top_logprobs=3`` with low-probability filler tokens."""
    entries = [(float(np.log(p)), token, None) for p, token in zip(probabilities, output, strict=True)]
    fillers = [(-20.0 - i, 7 - i, None) for i in range(3 - len(entries))]
    return {
        "output_token_logprobs": entries,
        "finish_reason": {"type": "stop"},
        **_flat_output_top_logprobs([[*entries, *fillers]] * len(output)),
    }


def _flat_output_top_logprobs(rows: list, *, null_prefix: int = 0) -> dict:
    """Encode nested [logprob, token_id, text] rows the way SGLang's return_flat_raw_output_top_logprobs does."""
    shape = [len(rows), len(rows[0]) if rows else 0]
    val = np.asarray([[entry[0] for entry in row] for row in rows], dtype="<f4").reshape(shape)
    idx = np.asarray([[entry[1] for entry in row] for row in rows], dtype="<i4").reshape(shape)
    return {
        "output_top_logprobs_shape": shape,
        "output_top_logprobs_null_prefix": null_prefix,
        "output_top_logprobs_val_flat_b64": pybase64.b64encode(val.tobytes()).decode("ascii"),
        "output_top_logprobs_idx_flat_b64": pybase64.b64encode(idx.tobytes()).decode("ascii"),
        "output_top_logprobs_val_flat_b64_dtype": "float32",
        "output_top_logprobs_idx_flat_b64_dtype": "int32",
    }


def _flat_sampling_mask(supports: list[list[int]], logprobs: list) -> dict:
    """Encode support rows and their log-probs (a value per row in selected mode, a row of values in support mode)
    the way SGLang's return_flat_raw_sampling_mask_b64 does."""
    flat_logprobs = [value for row in logprobs for value in (row if isinstance(row, list) else [row])]
    fields = {"output_token_sampling_mask_length": len(supports)}
    for key, values, dtype in (
        ("output_token_sampling_mask_flat", [token for row in supports for token in row], "<i4"),
        ("output_token_sampling_mask_lengths", [len(row) for row in supports], "<i4"),
        ("output_token_sampling_logprobs_flat", flat_logprobs, "<f4"),
    ):
        fields[f"{key}_b64"] = pybase64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")
        fields[f"{key}_b64_dtype"] = np.dtype(dtype).name
    return fields
