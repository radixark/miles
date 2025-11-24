import logging

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.core.models.gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source="MinimaxM2ForCausalLM", target=GPTModel)
class MinimaxM2Bridge(MegatronModelBridge):
    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        self._hf_config = hf_pretrained.config
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def mapping_registry(self) -> MegatronMappingRegistry:
        return TODO
