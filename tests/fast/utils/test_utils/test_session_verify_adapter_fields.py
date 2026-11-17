from scripts.tools.verify_session_tito_tokenizer import _runner_namespace
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace


class TestSessionVerifierAdapterFields:
    def test_adapter_keeps_each_sdk_owner_and_hook_value_when_building_the_local_namespace(self) -> None:
        """Each session benchmark setting keeps its declared SDK field and local wrapper-only values."""
        values = {
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen25",
            "context_length": 4096,
            "kv_cache_dtype": "fp8_e4m3",
            "mamba_full_memory_ratio": 0.3,
            "cuda_graph_max_bs_decode": 32,
            "cuda_graph_backend_prefill": "test-backend",
            "ep_size": 2,
        }
        generate = CustomFunctionConfig(
            path="test86.generate",
            config=ImmutableNamespace(session_verify_cycles=3, tool_call_failure_mode="append_user"),
        )
        agent = CustomFunctionConfig(path="test86.agent")
        config = parse_megatron_test_config().model_copy(
            update={
                "custom_generate_function_path": generate,
                "custom_agent_function_path": agent,
                "sglang": SglangConfig(models=[], base_args=values),
            }
        )
        before = dict(config)

        args = _runner_namespace(config=config, assistant_text_threshold=0.73, enable_spec=True)

        assert {name: vars(args)["sglang_" + name] for name in values} == values
        assert args.custom_generate_function_path == generate.path
        assert args.custom_agent_function_path == agent.path
        assert args.session_verify_cycles == 3
        assert args.tool_call_failure_mode == "append_user"
        assert args.assistant_text_threshold == 0.73
        assert args.enable_spec is True
        assert args.hf_checkpoint == config.hf_checkpoint
        assert args.rollout_batch_size == config.rollout_batch_size
        assert dict(config) == before
        args.hf_checkpoint = "/local-only"
        assert config.hf_checkpoint != args.hf_checkpoint
