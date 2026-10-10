"""Small deterministic samples for score-centering integration tests."""

from argparse import Namespace

import numpy as np
import pybase64

from miles.rollout.generate_utils.rollout_topk_logprobs import append_rollout_topk_logprobs
from miles.utils.types import Sample


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return str(tokens)


def _args(**overrides: object) -> Namespace:
    values = dict(
        ci_test=False,
        loss_type="score_centering",
        rollout_top_logprobs_num=3,
        rollout_sampling_logprobs_mode="selected",
        use_opd=False,
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
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        use_dynamic_global_batch_size=False,
        balance_data=False,
        qkv_format="thd",
        true_on_policy_mode=False,
        allgather_cp=False,
        log_probs_chunk_size=2,
        vocab_size=8,
        entropy_coef=0,
        observe_training_entropy=False,
        use_kl_loss=False,
    )
    return Namespace(**(values | overrides))


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
    rows = [[(logp, token, None) for logp, token in zip(logps, output, strict=True)]] * len(output)
    append_rollout_topk_logprobs(
        sample,
        {
            "output_token_logprobs": [(logp, token, None) for logp, token in zip(logps, output, strict=True)],
            "output_top_logprobs": rows,
        },
        3,
    )
    return sample


def _meta(output: list[int], probabilities: list[float]) -> dict:
    entries = [(float(np.log(p)), token, None) for p, token in zip(probabilities, output, strict=True)]
    return {
        "output_token_logprobs": entries,
        "output_top_logprobs": [entries] * len(output),
        "finish_reason": {"type": "stop"},
    }


def _flat_meta(output: list[int], probabilities: list[float]) -> dict:
    """``_meta`` with the candidates in SGLang's flat base64 output top-logprob format."""
    meta = _meta(output, probabilities)
    return meta | _flat_output_top_logprobs(meta.pop("output_top_logprobs"))


def _flat_output_top_logprobs(rows: list, *, null_prefix: int = 0, b64: bool = True) -> dict:
    """Encode nested [logprob, token_id, text] rows the way SGLang's return_flat_raw_output_top_logprobs does."""
    shape = [len(rows), len(rows[0]) if rows else 0]
    val = np.asarray([[entry[0] for entry in row] for row in rows], dtype=np.float32).reshape(shape)
    idx = np.asarray([[entry[1] for entry in row] for row in rows], dtype=np.int32).reshape(shape)
    fields = {"output_top_logprobs_shape": shape, "output_top_logprobs_null_prefix": null_prefix}
    if not b64:
        return fields | {
            "output_top_logprobs_val_flat": val.ravel().tolist(),
            "output_top_logprobs_idx_flat": idx.ravel().tolist(),
        }
    return fields | {
        "output_top_logprobs_val_flat_b64": pybase64.b64encode(val.tobytes()).decode("ascii"),
        "output_top_logprobs_idx_flat_b64": pybase64.b64encode(idx.tobytes()).decode("ascii"),
        "output_top_logprobs_val_flat_b64_dtype": "float32",
        "output_top_logprobs_idx_flat_b64_dtype": "int32",
    }
