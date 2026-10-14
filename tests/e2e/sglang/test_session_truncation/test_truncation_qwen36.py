import pytest
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role.test_qwen36 import CONFIG
from tests.e2e.sglang.test_session_truncation._common import CASE_IDS, CASES, run_case, serve_model

register_cuda_ci(
    est_time=1400, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"], num_gpus=2
)


@pytest.fixture(scope="module")
def engine():
    with serve_model(CONFIG) as engine:
        yield engine


@pytest.mark.parametrize(("version", "truncate"), CASES, ids=CASE_IDS)
def test_qwen36(engine, version, truncate):
    server, model_path = engine
    run_case(server, model_path, CONFIG, version=version, truncate=truncate)
