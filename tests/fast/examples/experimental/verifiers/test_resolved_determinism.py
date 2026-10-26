import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from examples.experimental.verifiers import verifiers_rollout

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.utils.args.custom_view import ImmutableNamespace


class TestResolvedDeterminism:
    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("evaluation", [False, True])
    async def test_task_groups_apply_resolved_server_determinism_to_each_selected_context(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool, evaluation: bool
    ) -> None:
        """Resolved deterministic inference assigns per-rollout seeds while retaining the selected train or eval client."""
        adapter = object.__new__(verifiers_rollout.VerifiersRolloutFn)
        adapter.args = ImmutableNamespace(
            sglang=SglangConfig(models=[], base_args={"enable_deterministic_inference": enabled})
        )
        adapter.model = "policy"
        sampling = ImmutableNamespace(temperature=0.7, sampling_seed=None)
        adapter.ctx = SimpleNamespace(client=object(), sampling=sampling)
        eval_ctx = SimpleNamespace(client=object(), sampling=sampling)
        ctx = eval_ctx if evaluation else adapter.ctx
        rollouts = [SimpleNamespace(ctx=ctx), SimpleNamespace(ctx=ctx)]
        semaphore = asyncio.Semaphore(1)
        task = object()
        expected_result = object()

        async def run(received: asyncio.Semaphore) -> Any:
            assert received is semaphore
            return expected_result

        def episode(received_task: object, received_ctx: object, *, n: int) -> SimpleNamespace:
            assert received_task is task
            assert received_ctx is ctx
            assert n == 2
            return SimpleNamespace(rollouts=rollouts, run=run)

        adapter.env = SimpleNamespace(episode=episode)
        monkeypatch.setattr(
            verifiers_rollout, "_import_verifiers", lambda: SimpleNamespace(ModelContext=SimpleNamespace)
        )
        result = await adapter._run_task_group(
            task, n=2, semaphore=semaphore, seed_base=101, ctx=eval_ctx if evaluation else None
        )

        assert result is expected_result
        assert [rollout.ctx.sampling.sampling_seed for rollout in rollouts] == (
            [101, 102] if enabled else [None, None]
        )
        assert all(rollout.ctx.client is ctx.client for rollout in rollouts)
        assert sampling.sampling_seed is None
        assert sampling.temperature == 0.7


class TestConstructorSamplingOwnership:
    def test_actual_constructor_keeps_eval_minimum_tokens_out_of_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructor wiring applies minimum output length only to evaluation sampling."""
        from miles.rollout.base_types import RolloutFnConstructorInput

        config_path = tmp_path / "environment.toml"
        config_path.write_text('[harness]\nid = "test"\n')
        monkeypatch.setenv("VERIFIERS_CONFIG", str(config_path))
        args = ImmutableNamespace(
            rollout_global_dataset=False,
            partial_rollout=False,
            multimodal_keys=None,
            chat_template_path=None,
            use_opd=False,
            use_rollout_routing_replay=False,
            use_rollout_indexer_replay=False,
            hf_checkpoint="policy",
            rollout_temperature=0.7,
            rollout_top_p=0.9,
            rollout_top_k=None,
            rollout_max_response_len=8,
            apply_chat_template_kwargs={},
            eval_temperature=None,
            eval_top_p=None,
            eval_top_k=None,
            eval_max_response_len=12,
            eval_max_context_len=None,
            eval_max_prompt_len=None,
            eval_reward_key=None,
            reward_key=None,
            eval_min_new_tokens=9,
            inference_runtime_mut_state=SimpleNamespace(engine_count=2),
            sglang_server_concurrency=1,
            sglang=SglangConfig(models=[], base_args={"tokenizer_path": None}),
        )

        class SdkReached(Exception):
            pass

        class TrainClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise SdkReached

        runtime = SimpleNamespace(
            EnvConfig=SimpleNamespace(
                model_validate=lambda value: SimpleNamespace(is_legacy=False, harness=SimpleNamespace(id="test"))
            ),
            Environment=lambda config: object(),
            SamplingConfig=ImmutableNamespace,
            TrainClient=TrainClient,
        )
        monkeypatch.setattr(verifiers_rollout, "_import_verifiers", lambda: runtime)
        renderers = ModuleType("renderers")
        renderers.base = ModuleType("renderers.base")
        renderers.base.MODEL_RENDERER_MAP = {}
        monkeypatch.setitem(sys.modules, "renderers", renderers)
        monkeypatch.setitem(sys.modules, "renderers.base", renderers.base)
        adapter = object.__new__(verifiers_rollout.VerifiersRolloutFn)
        with pytest.raises(SdkReached):
            adapter.__init__(RolloutFnConstructorInput(args=args, data_source=object()))

        assert "min_tokens" not in dict(adapter.sampling)
        assert adapter.eval_sampling.min_tokens == 9
        assert adapter.sampling.max_tokens == 8
        assert adapter.eval_sampling.max_tokens == 12
        assert "rollout_min_new_tokens" not in dict(args)
