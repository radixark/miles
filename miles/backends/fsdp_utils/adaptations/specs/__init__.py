"""Per-arch ``ArchAdapter``s — the one place a new architecture plugs into the FSDP backend.

To add an arch, create ``specs/<arch>.py`` with an ``ArchAdapter`` subclass that overrides only the hooks
it needs (see ``arch_adapter.py``), and list it in ``_ADAPTERS`` below. An architecture without an adapter
runs on the stock HF path.
"""

from types import MappingProxyType

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.specs.glm4_moe_lite import Glm4MoeLiteAdapter
from miles.backends.fsdp_utils.adaptations.specs.nemotron_h import NemotronHAdapter
from miles.backends.fsdp_utils.adaptations.specs.qwen3 import Qwen3Adapter
from miles.backends.fsdp_utils.adaptations.specs.qwen3_5 import Qwen35Adapter, Qwen35MoeAdapter
from miles.backends.fsdp_utils.adaptations.specs.qwen3_moe import Qwen3MoeAdapter
from miles.backends.fsdp_utils.adaptations.specs.qwen3_vl import Qwen3VLAdapter

_ADAPTERS: tuple[ArchAdapter, ...] = (
    Glm4MoeLiteAdapter(),
    NemotronHAdapter(),
    Qwen3Adapter(),
    Qwen35Adapter(),
    Qwen35MoeAdapter(),
    Qwen3MoeAdapter(),
    Qwen3VLAdapter(),
)
_ADAPTER_BY_MODEL_TYPE = MappingProxyType(
    {model_type: adapter for adapter in _ADAPTERS for model_type in adapter.model_types}
)

_STOCK_HF = ArchAdapter()


def resolve_arch_adapter(hf_config) -> ArchAdapter:
    return _ADAPTER_BY_MODEL_TYPE.get(getattr(hf_config, "model_type", None), _STOCK_HF)
