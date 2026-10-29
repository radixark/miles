"""Every miles process imports the Muse Spark HF config (miles.utils.hf_utils.config registers it), so that
import must not load Megatron Core: its Transformer Engine dlopens libcuda.so.1 and fails on a CPU-only host."""

import pytest

from tests.fast.import_isolation_utils import modules_imported_by


@pytest.mark.parametrize(
    "module",
    ["miles.utils.hf_utils.config", "miles_plugins.models.muse_spark.hf_config", "miles.rollout"],
)
def test_importing_the_config_path_loads_no_megatron(module):
    loaded = modules_imported_by(module)
    assert not sorted(name for name in loaded if name == "megatron" or name.startswith("megatron."))
