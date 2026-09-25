"""Small deterministic samples for score-centering integration tests."""

from argparse import Namespace

import numpy as np

from miles.rollout.generate_utils.score_centering import append_score_centering_topk
from miles.utils.types import Sample


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return str(tokens)


def _args(**overrides: object) -> Namespace:
    values = dict(
        loss_type="score_centering",
        score_centering_top_k=3,
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        rollout_temperature=0.7,
        rollout_top_p=1.0,
        rollout_top_k=-1,
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
    append_score_centering_topk(
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
