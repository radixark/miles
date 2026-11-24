import asyncio
import threading
import urllib.request
from pathlib import Path
from uuid import uuid4

import chz
from tinker_agent.harbor import HarborConfig, HarborDatasetBuilder, HarborRolloutStrategy
from tinker_cookbook.capture.store import capture_to_store
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl import train


def keep_store_alive(base_url, stop, interval_sec=300.0):
    # capture_to_store holds no lease on its daemon, which exits after an idle hour; a run can wait
    # longer than that for a shared trainer without sampling anything.
    while not stop.wait(interval_sec):
        urllib.request.urlopen(f"{base_url}/touch", timeout=10).close()


def main(config: train.Config, harbor: HarborConfig):
    if config.rollout_error_tolerance is not None:
        raise ValueError("This recipe owns config.rollout_error_tolerance through HarborRolloutStrategy")
    # External agents own sampling parameters; the official proxy has no temperature-default hook.
    if config.temperature != 1.0:
        raise ValueError("Set temperature in the Harbor agent config; config.temperature must remain 1.0")
    renderer_name = config.renderer_name or get_recommended_renderer_name(config.model_name)
    trial_template = Path(harbor.trial_config).read_text()
    run_id = uuid4().hex
    with capture_to_store(run_id, data_dir=Path(config.log_path) / "capture") as captured:
        stop = threading.Event()
        threading.Thread(target=keep_store_alive, args=(captured.base_url, stop), daemon=True).start()
        config = chz.replace(
            config,
            renderer_name=renderer_name,
            rollout_error_tolerance=HarborRolloutStrategy(
                renderer_name=renderer_name,
                trial_template=trial_template,
                store_url=captured.base_url,
                run_id=run_id,
                harbor=harbor,
            ),
        )
        try:
            asyncio.run(train.main(config))
        finally:
            stop.set()


@chz.blueprint.exit_on_entrypoint_error
def cli(argv=None):
    return (
        chz.Blueprint(main)
        .apply({"config.recipe_name": "harbor-tinker", "config.dataset_builder": HarborDatasetBuilder})
        .make_from_argv(argv)
    )


if __name__ == "__main__":
    cli()
