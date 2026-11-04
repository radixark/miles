from typing import Any

import pytest

from examples.experimental.multi_agent import rollout_with_multi_agents
from examples.experimental.multi_agent.agent_system import AgentState
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


class TestMultiAgentRolloutWrapper:
    @pytest.mark.parametrize("evaluation,expected_context", [(False, 32), (True, 12)])
    async def test_wrapper_preserves_caller_configuration_and_isolates_each_agent_state(
        self, monkeypatch: pytest.MonkeyPatch, evaluation: bool, expected_context: int
    ) -> None:
        """Train and eval calls preserve caller state while forwarding immutable config and isolated agent inputs."""
        args = ImmutableNamespace(
            hf_checkpoint="model", chat_template_path="template", rollout_max_context_len=32, eval_max_context_len=12
        )
        original = dict(args)
        sampling_params = {"max_new_tokens": 9, "nested": {"temperatures": [0.7]}}
        tokenizer = object()
        tokenizer_calls: list[dict[str, Any]] = []
        states: list[AgentState] = []
        returned: list[list[Sample]] = []

        def load_tokenizer(checkpoint: str, *, chat_template_path: str, trust_remote_code: bool) -> object:
            tokenizer_calls.append(
                dict(checkpoint=checkpoint, chat_template_path=chat_template_path, trust_remote_code=trust_remote_code)
            )
            return tokenizer

        async def run_agents(config: ImmutableNamespace, *, state: AgentState) -> list[Sample]:
            assert isinstance(config, ImmutableNamespace)
            assert config.rollout_max_context_len == expected_context
            assert config.num_parallel == 5
            assert config.incorrect_reward_weight == 0.8
            assert config.correct_reward_weight == 1.2
            assert state.tokenizer is tokenizer
            assert state.sampling_params == sampling_params
            assert state.sampling_params is not sampling_params
            assert state.sampling_params["nested"] is not sampling_params["nested"]
            assert state.results == {"solver": [], "rewriter": [], "selector": []}
            states.append(state)
            state.sampling_params["nested"]["temperatures"].append(0.1)
            samples = [Sample(prompt="first-result"), Sample(prompt="second-result")]
            returned.append(samples)
            return samples

        monkeypatch.setattr(rollout_with_multi_agents, "load_tokenizer", load_tokenizer)
        samples = [Sample(prompt="first-input"), Sample(prompt="second-input")]
        outputs = []
        path = rollout_with_multi_agents.MULTI_AGENT_CONFIGS["custom_multi_agent_function_path"]
        with function_registry.temporary(path, run_agents):
            for sample in samples:
                outputs.append(
                    await rollout_with_multi_agents.generate_with_multi_agents(
                        args=args, sample=sample, sampling_params=sampling_params, evaluation=evaluation
                    )
                )

        assert states[0] is not states[1]
        assert states[0].results is not states[1].results
        assert states[0].sample is samples[0]
        assert states[1].sample is samples[1]
        assert outputs[0] is returned[0]
        assert outputs[1] is returned[1]
        assert all(
            sorted(sample.prompt for sample in output) == ["first-result", "second-result"] for output in outputs
        )
        assert tokenizer_calls == [dict(checkpoint="model", chat_template_path="template", trust_remote_code=True)] * 2
        assert sampling_params == {"max_new_tokens": 9, "nested": {"temperatures": [0.7]}}
        assert dict(args) == original
