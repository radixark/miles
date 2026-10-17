from pathlib import Path

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.recipes.gsm8k import Gsm8kRun


async def run_hot_restart_soak(
    *,
    run: Gsm8kRun,
    runner_config: SoakRunnerConfig,
    evidence_dir: Path,
) -> SoakRunner:
    raise NotImplementedError
