from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.utils import arguments


def _args(**overrides) -> Namespace:
    values = dict(
        eval_pool_num_engines=2,
        fully_async=True,
        colocate=False,
        eval_num_gpus=0,
        eval_interval=10,
        eval_max_in_flight=1,
        ft_components=[],
        rollout_external=False,
        rollout_num_gpus_per_engine=1,
        update_weights_interval=1,
    )
    return Namespace(**(values | overrides))


@pytest.fixture(autouse=True)
def _internal_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arguments, "_compute_rollout_external", lambda args: args.rollout_external)


class TestValidateEvalPool:
    def test_a_supported_combination_passes(self):
        arguments._validate_eval_pool(_args(), checkpoint_backend=False)

    def test_zero_disables_every_check(self):
        arguments._validate_eval_pool(_args(eval_pool_num_engines=0, fully_async=False), checkpoint_backend=False)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            (dict(eval_pool_num_engines=-1), ">= 0"),
            (dict(fully_async=False), "--fully-async"),
            (dict(colocate=True), "--colocate"),
            (dict(eval_num_gpus=2), "--eval-num-gpus"),
            (dict(rollout_external=True), "external engines"),
            (dict(eval_interval=None), "--eval-interval"),
            (dict(eval_max_in_flight=2), "--eval-max-in-flight 1"),
            (dict(ft_components=["rollout"]), "fault tolerance"),
            (dict(update_weights_interval=2), "--update-weights-interval 1"),
        ],
    )
    def test_an_unsupported_combination_is_rejected_with_its_reason(self, overrides, message):
        with pytest.raises(AssertionError, match=message):
            arguments._validate_eval_pool(_args(**overrides), checkpoint_backend=False)

    def test_a_checkpoint_backend_is_rejected(self):
        with pytest.raises(AssertionError, match="CheckpointEvalFn"):
            arguments._validate_eval_pool(_args(), checkpoint_backend=True)


def _sglang(*, num_models=1, pd=False, gpus_per_engine=1, num_engines=4) -> tuple:
    models = [
        SimpleNamespace(
            name=f"model-{i}",
            has_pd_disaggregation=pd,
            server_groups=[SimpleNamespace(num_gpus_per_engine=gpus_per_engine)],
            num_server_cells=lambda scaling: num_engines,
        )
        for i in range(num_models)
    ]
    return SimpleNamespace(models=models), SimpleNamespace()


class TestCheckEvalPoolEngines:
    def test_a_single_model_with_spare_engines_passes(self):
        sglang, scaling = _sglang()
        arguments._check_eval_pool_engines(_args(), sglang=sglang, sglang_scaling=scaling)

    @pytest.mark.parametrize(
        ("sglang_kwargs", "message"),
        [
            (dict(num_models=2), "several served models"),
            (dict(pd=True), "prefill/decode"),
            (dict(gpus_per_engine=2), "--rollout-num-gpus-per-engine"),
            (dict(num_engines=2), "leaves training none"),
        ],
    )
    def test_an_unsupported_engine_layout_is_rejected(self, sglang_kwargs, message):
        sglang, scaling = _sglang(**sglang_kwargs)
        with pytest.raises(AssertionError, match=message):
            arguments._check_eval_pool_engines(_args(), sglang=sglang, sglang_scaling=scaling)
