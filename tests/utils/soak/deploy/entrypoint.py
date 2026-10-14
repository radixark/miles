from pathlib import Path

from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.entrypoint import run_soak
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.utils import compute_release_of_config
from tests.utils.soak.deploy.actions.hot_restart import HotRestartForm
from tests.utils.soak.deploy.observers import DeploymentObserver
from tests.utils.soak.deploy.session import LauncherChain, execute_hot_restart_session
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND
from tests.utils.soak.deploy.utils import compute_checkpoint_dir
from tests.utils.soak.recipes.gsm8k import Gsm8kRun


async def run_hot_restart_soak(
    *,
    run: Gsm8kRun,
    runner_config: SoakRunnerConfig,
    evidence_dir: Path,
) -> SoakRunner:
    config = run.launch_spec.config
    chain = LauncherChain()
    forms = {
        DEPLOYMENT_TARGET_KIND: [
            HotRestartForm(
                launch_spec=run.launch_spec,
                event_log=run.event_log,
                chain=chain,
            )
        ]
    }

    return await run_soak(
        config=config,
        dump_dir=Path(run.dump_dir),
        sut_run=execute_hot_restart_session(run, chain=chain),
        runner_config=runner_config,
        forms=forms,
        event_log=run.event_log,
        observer=DeploymentObserver(
            namespace=config.namespace,
            release=compute_release_of_config(config),
            trainer_id=DEFAULT_TRAINER_ID,
            checkpoint_dir=compute_checkpoint_dir(run.dump_dir),
            events_dir=run.events_dir,
        ),
        evidence_dir=evidence_dir,
    )
