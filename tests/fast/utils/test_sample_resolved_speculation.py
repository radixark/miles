from types import SimpleNamespace

import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.utils.types import Sample


class TestSampleResolvedSpeculation:
    @pytest.mark.parametrize("algorithm", [None, "EAGLE"])
    def test_sample_metadata_accumulates_speculative_counters_only_for_the_resolved_algorithm(
        self, algorithm: str | None
    ) -> None:
        """Resolved speculation gates additive counters while ordinary status and prefix metadata remain active."""
        args = SimpleNamespace(sglang=make_sglang_config(speculative_algorithm=algorithm))
        sample = Sample(tokens=[1, 2, 3, 4, 5])
        meta = {
            "finish_reason": {"type": "stop"},
            "spec_verify_ct": 2,
            "spec_num_correct_drafts": 3,
            "spec_num_proposed_drafts": 4,
            "completion_tokens": 5,
            "prompt_tokens": 10,
            "cached_tokens": 2,
        }

        sample.update_from_meta_info(args, meta)
        sample.update_from_meta_info(args, meta)

        assert sample.spec_info == (Sample.SpecInfo(6, 8, 4, 10) if algorithm else Sample.SpecInfo())
        assert sample.prefix_cache_info.cached_tokens == 4
        assert sample.prefix_cache_info.total_prompt_tokens == 20
        assert sample.status == Sample.Status.COMPLETED
