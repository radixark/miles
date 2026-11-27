import os
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class KimiCodeAgent(BaseInstalledAgent):
    SUPPORTS_ATIF = False

    @staticmethod
    @override
    def name() -> str:
        return "kimi-code"

    @override
    def get_version_command(self) -> str:
        return 'export PATH="$HOME/.kimi-code/bin:$HOME/.local/bin:$PATH"; kimi --version'

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl ca-certificates",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash && "
                'export PATH="$HOME/.kimi-code/bin:$HOME/.local/bin:$PATH" && '
                "kimi --version"
            ),
        )

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self._parsed_model_provider != "openai":
            raise ValueError("KimiCodeAgent requires an OpenAI-compatible model provider")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required")
        for name in ("KIMI_MODEL_BASE_URL", "KIMI_MODEL_MAX_CONTEXT_SIZE", "KIMI_MODEL_THINKING_EFFORT"):
            if not os.environ.get(name):
                raise ValueError(f"{name} is required")

        env = {
            "HARBOR_KIMI_INSTRUCTION": self.render_instruction(instruction),
            "KIMI_CODE_HOME": "/logs/agent/kimi-code",
            "KIMI_CODE_NO_AUTO_UPDATE": "1",
            "KIMI_CODE_TRUST_WORKSPACE": "1",
            "KIMI_DISABLE_TELEMETRY": "1",
            "KIMI_MODEL_NAME": self._parsed_model_name,
            "KIMI_MODEL_PROVIDER_TYPE": "openai",
            "KIMI_MODEL_BASE_URL": os.environ["KIMI_MODEL_BASE_URL"],
            "KIMI_MODEL_MAX_CONTEXT_SIZE": os.environ["KIMI_MODEL_MAX_CONTEXT_SIZE"],
            "KIMI_MODEL_CAPABILITIES": "thinking,tool_use",
            "KIMI_MODEL_THINKING_EFFORT": os.environ["KIMI_MODEL_THINKING_EFFORT"],
            "CI": "1",
        }
        command = (
            'export PATH="$HOME/.kimi-code/bin:$HOME/.local/bin:$PATH"; '
            'export KIMI_MODEL_API_KEY="$OPENAI_API_KEY"; '
            "mkdir -p /logs/agent/kimi-code; "
            'kimi -p "$HARBOR_KIMI_INSTRUCTION" --output-format stream-json '
            "2>/logs/agent/kimi-code.stderr.log | tee /logs/agent/kimi-code.jsonl"
        )
        await self.exec_as_agent(environment, command=command, env=env, cwd="/app")
