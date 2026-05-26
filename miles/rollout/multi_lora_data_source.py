import copy
import logging
from argparse import Namespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import ray

from miles.ray.multi_lora_controller import get_multi_lora_controller
from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.adapter_config import AdapterConfig, AdapterState, RegisteredAdapter
from miles.utils.types import AdapterRef, RewardSpec, Sample

logger = logging.getLogger(__name__)

# Cap parallelism for adapter source creation. Each worker tokenizes/renders a full
# dataset; with too many threads the GIL/IO contention outweighs the speedup.
_MAX_RECONCILE_WORKERS = 16


def fetch_active_adapters() -> dict[str, RegisteredAdapter]:
    return ray.get(get_multi_lora_controller().active_adapters.remote())


def fetch_adapter_steps() -> dict[str, int]:
    return ray.get(get_multi_lora_controller().adapter_train_steps.remote())


class MultiLoRADataSource(DataSource):
    def __init__(self, args: Namespace):
        self.args = args
        self.sources: dict[str, RolloutDataSource] = {}
        self.configs: dict[str, AdapterConfig] = {}

        self.source_queue = deque()
        self._reconcile(fetch_active_adapters())

    def _update_source_queue(self, active_names):
        # Filter out any adapter names that are gone while retaining order
        new_source_queue = deque()

        # Keep old entries in order and add into new queue
        in_queue = set()
        while self.source_queue:
            if (name := self.source_queue.popleft()) in active_names:
                new_source_queue.append(name)
                in_queue.add(name)

        # Add new entries to the end
        for name in active_names:
            if name not in in_queue:
                new_source_queue.append(name)

        assert set(new_source_queue) == active_names and len(new_source_queue) == len(active_names)

        self.source_queue = new_source_queue

    def _reconcile(self, adapters: dict[str, RegisteredAdapter]) -> None:
        # Clean up old sources
        for name in list(self.sources):
            if name not in adapters:
                del self.sources[name]
                del self.configs[name]
                logger.info(f"Removed data source for adapter '{name}'")

        # Collect adapters that need new data sources. Each source spins up a fresh
        # Dataset which reads + tokenizes the prompt file, so this is the slow path
        # we parallelize below.
        pending = [(name, adapter) for name, adapter in adapters.items() if name not in self.sources]
        if pending:
            # Fetch steps once for the whole batch instead of one Ray RPC per adapter.
            steps = fetch_adapter_steps()
            workers = min(_MAX_RECONCILE_WORKERS, len(pending))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="miles-datasource") as ex:
                    built = list(
                        ex.map(
                            lambda na: (na[0], self._create_adapter_source(na[0], na[1].config, steps)),
                            pending,
                        )
                    )
            else:
                built = [(name, self._create_adapter_source(name, adapter.config, steps)) for name, adapter in pending]
            for name, source in built:
                self.sources[name] = source
                logger.info(f"Created data source for adapter '{name}' from {adapters[name].config.data}")

        for name, adapter in adapters.items():
            self.configs[name] = adapter.config

    def _create_adapter_source(
        self, name: str, config: AdapterConfig, steps: dict[str, int]
    ) -> RolloutDataSource:
        adapter_args = copy.copy(self.args)

        # Data
        adapter_args.prompt_data = config.data
        adapter_args.input_key = config.input_key or self.args.input_key
        adapter_args.label_key = config.label_key or self.args.label_key
        adapter_args.metadata_key = config.metadata_key or self.args.metadata_key

        # Checkpointing
        adapter_args.save = config.dir or self.args.save
        adapter_args.load = config.dir or self.args.load
        adapter_args.start_rollout_id = steps.get(name, 0)

        return RolloutDataSource(adapter_args)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Runs a round robin around the data sources and preserves the round robin ordering
        even when new datasources are added or removed.
        """
        adapters = fetch_active_adapters()
        self._reconcile(adapters)

        active_names = set(n for n in self.sources if adapters[n].state == AdapterState.RUNNING)
        datasource_drained = set(n for n in self.sources if adapters[n].state == AdapterState.DRAINING_DATASOURCE)

        assert len(active_names) + len(datasource_drained) > 0, "get_samples called without any active adapters"

        # Run one last iter for those being drained, since sglang rollout needs to be able to run one last
        # time for the adapter after the draining has kicked off in order to update the adapter states
        active_names |= datasource_drained
        self._update_source_queue(active_names)

        refs = {name: AdapterRef(name=name, slot=adapters[name].slot) for name in active_names}
        reward_specs = {
            name: RewardSpec(
                rm_type=adapters[name].config.rm_type,
                custom_rm_path=adapters[name].config.custom_rm_path,
            )
            for name in active_names
        }

        samples_per_adapter, remainder = divmod(num_samples, len(self.source_queue))

        # Get samples from each data source
        all_samples: list[list[Sample]] = []
        for i in range(len(self.source_queue)):
            # for samples 0 -> remainder, add an extra sample
            extra = int(i < remainder)
            samples_needed = samples_per_adapter + extra

            # If no samples needed, then exit early
            if samples_needed == 0:
                break

            name = self.source_queue.popleft()
            config = adapters[name].config
            # Add the name back into the queue for next time get_samples is called, preserving
            # round robin ordering
            self.source_queue.append(name)

            source: RolloutDataSource = self.sources[name]
            adapter_samples = source.get_samples(samples_needed)

            # Add LoRA adapter data + per adapter reward fn data, also merge adapter meta
            ref = refs[name]
            reward_spec = reward_specs[name]
            for group in adapter_samples:
                for sample in group:
                    sample.adapter = ref
                    sample.reward_spec = reward_spec
                    sample.metadata = {**config.metadata, **sample.metadata}
            all_samples.extend(adapter_samples)

            # Begin deregistration process when out of data
            # sample_group_index is the same as tracking the row index
            # Default to length of dataset, override if num rollout is set
            default_num_row = (getattr(config, "num_epoch", 1) or 1) * len(source.dataset)
            num_row = config.num_row or default_num_row
            if source.sample_group_index >= num_row:
                logger.info(f"Adapter '{name}' reached num_row={num_row}, deregistering")
                datasource_drained.add(name)

        if datasource_drained:
            ray.get(
                get_multi_lora_controller().update_adapter_state.remote(
                    list(datasource_drained), AdapterState.DRAINING_INFLIGHT
                )
            )

        # Verify we always get num_samples at the end
        assert len(all_samples) == num_samples

        return all_samples

    def add_samples(self, samples: list[list[Sample]]):
        """Re-queue retried groups; drops groups for non-RUNNING adapters."""
        adapters = fetch_active_adapters()
        self._reconcile(adapters)

        for group in samples:
            name = group[0].adapter.name if group and group[0].adapter else None
            if not name or name not in self.sources:
                continue
            adapter = adapters.get(name)
            if adapter is None or adapter.state != AdapterState.RUNNING:
                continue
            self.sources[name].add_samples([group])

    # TODO: support save/loading per adapter
    # Saving and loading both currently don't work for a single adapter,
    # so this functionality doesn't directly work right now
    def save(self, rollout_id):
        # Note: the rollout_id is unused for multilora in favor
        # of the actual last train step that was tracked for that lora
        steps = fetch_adapter_steps()

        for adapter_name, source in self.sources.items():
            step = steps.get(adapter_name, 0)
            source.save(step)

    def load(self, rollout_id=None):
        steps = fetch_adapter_steps()

        for adapter_name, source in self.sources.items():
            step = steps.get(adapter_name, 0)
            source.load(step)
