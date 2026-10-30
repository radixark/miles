from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.prefill_logprobs import (
    recompute_rollout_logprobs_via_prefill,
    recompute_samples_rollout_logprobs_via_prefill,
)
from miles.utils.types import Sample


class TestDisabledPrefillContract:
    @pytest.mark.parametrize("batch", [False, True])
    async def test_disabled_recompute_preserves_native_scores_without_network(
        self, monkeypatch: pytest.MonkeyPatch, batch: bool
    ) -> None:
        """Disabling prefill preserves native scores and never flushes or scores a cache."""

        async def unexpected_post(*args: object, **kwargs: object) -> None:
            pytest.fail("Disabled recompute must not contact the engine")

        monkeypatch.setattr("miles.rollout.generate_utils.prefill_logprobs.post", unexpected_post)
        sample = Sample(tokens=[1, 2], response_length=1, rollout_log_probs=[-0.25], metadata={"native": True})
        args = SimpleNamespace(recompute_logprobs_via_prefill=False)

        if batch:
            await recompute_samples_rollout_logprobs_via_prefill(
                args, [sample], url="http://engine/generate", sampling_params={}
            )
        else:
            await recompute_rollout_logprobs_via_prefill(
                args, sample, url="http://engine/generate", sampling_params={}
            )

        assert sample.rollout_log_probs == [-0.25]
        assert sample.metadata == {"native": True}
