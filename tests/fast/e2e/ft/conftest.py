import pytest
from tests.e2e.ft.conftest_ft import comparisons


@pytest.fixture
def comparison_primitives_but_reconfigure(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    called: list[str] = []
    for name in (
        "assert_metrics_classified",
        "compare_metrics",
        "compare_dumps",
        "compare_inference_engine_checksums",
        "assert_engine_weights_moved",
        "assert_gradients_nonzero",
    ):
        monkeypatch.setattr(comparisons, name, lambda *args, _name=name, **kwargs: called.append(_name))
    return called
