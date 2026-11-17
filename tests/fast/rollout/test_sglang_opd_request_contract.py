from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults, resolve_parse_boundary_configs

from miles.rollout import sglang_rollout
from miles.utils.types import Sample


class TestSglangOpdRequestContract:
    @pytest.mark.parametrize(
        "enabled,topk,strategy,expected",
        [
            (False, 3, "only-student", False),
            (True, 0, "only-student", False),
            (True, 3, "only-teacher", False),
            (True, 3, "only-student", True),
            (True, 3, "intersection", True),
        ],
    )
    async def test_student_topk_request_and_capture_follow_the_opd_strategy(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool, topk: int, strategy: str, expected: bool
    ) -> None:
        """Only strategies needing student candidates request and retain their top-k probabilities."""
        monkeypatch.setattr(sglang_rollout.SingletonMeta, "_instances", {})
        monkeypatch.setattr(
            sglang_rollout,
            "load_tokenizer",
            lambda *args, **kwargs: SimpleNamespace(encode=lambda prompt, add_special_tokens: [1, 2]),
        )
        monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
        payloads: list[dict] = []
        candidate = [[-0.1, 3], [-1.2, 4], [-2.3, 5]]

        async def fake_post(url: str, payload: dict, headers: dict | None = None) -> dict:
            payloads.append(payload)
            return {"text": "", "meta_info": {"finish_reason": {"type": "stop"}, "output_top_logprobs": [candidate]}}

        monkeypatch.setattr(sglang_rollout, "post", fake_post)
        args = resolve_parse_boundary_configs(
            Namespace(
                **(
                    parser_defaults()
                    | dict(
                        ci_test=False,
                        sglang_router_ip="router",
                        sglang_router_port=30000,
                        use_opd=enabled,
                        opd_log_prob_top_k=topk,
                        opd_top_k_strategy=strategy,
                        rollout_top_logprobs_num=0,
                        use_rollout_indexer_replay=True,
                    )
                )
            )
        )
        sample = Sample(prompt="p", metadata={"preserved": True})

        await sglang_rollout.generate(args, sample, {"max_new_tokens": 4})

        assert len(payloads) == 1
        assert payloads[0]["return_indexer_topk"] is True
        assert ("top_logprobs_num" in payloads[0]) is expected
        assert sample.metadata["preserved"] is True
        if expected:
            assert payloads[0]["top_logprobs_num"] == 3
            assert sample.metadata["opd_student_top_logprobs"] == [candidate]
        else:
            assert "opd_student_top_logprobs" not in sample.metadata
