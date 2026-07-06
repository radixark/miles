"""Multi-LoRA data source for the fully-async design.

Reads the new controller's ``active_adapters`` (RegisteredAdapter views,
state=RUNNING — no state machine), round-robins over per-adapter prompt
sources, tags samples with ``AdapterRef``/``RewardSpec``, and calls
``controller.deregister_adapter`` when an adapter reaches its ``num_row``.
Recycles aborted/dummied groups back to the per-adapter source. No drain states.
"""

import copy
import logging
from argparse import Namespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import ray

from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.adapter_config import RegisteredAdapter
from miles.utils.types import AdapterRef, RewardSpec, Sample

from miles.ray.multi_lora_controller import get_multi_lora_controller

logger = logging.getLogger(__name__)

MAX_RECONCILE_WORKERS = 16


def fetch_active_adapters() -> dict[str, RegisteredAdapter]:
    return ray.get(get_multi_lora_controller().active_adapters.remote())


class MultiLoRAAsyncDataSource(DataSource):
    def __init__(self, args: Namespace):
        self.args = args
        self.sources: dict[str, RolloutDataSource] = {}
        self.source_queue: deque = deque()
        self.reconcile(fetch_active_adapters())

    def reconcile(self, adapters: dict[str, RegisteredAdapter]) -> None:
        for name in list(self.sources):
            if name not in adapters:
                del self.sources[name]
                logger.info(f"Removed data source for adapter '{name}'")
        pending = [(name, a) for name, a in adapters.items() if name not in self.sources]
        if pending:
            workers = min(MAX_RECONCILE_WORKERS, len(pending))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mlora-ds") as ex:
                    built = list(ex.map(lambda na: (na[0], self.create_source(na[1])), pending))
            else:
                built = [(name, self.create_source(a)) for name, a in pending]
            for name, source in built:
                self.sources[name] = source
                logger.info(f"Created data source for adapter '{name}'")
        self.update_queue(set(adapters))

    def create_source(self, adapter: RegisteredAdapter) -> RolloutDataSource:
        config = adapter.config
        adapter_args = copy.copy(self.args)
        adapter_args.prompt_data = config.data
        adapter_args.input_key = config.input_key or self.args.input_key
        adapter_args.label_key = config.label_key or self.args.label_key
        adapter_args.metadata_key = config.metadata_key or self.args.metadata_key
        adapter_args.save = config.save or self.args.save
        adapter_args.load = config.save or self.args.load
        adapter_args.start_rollout_id = 0
        return RolloutDataSource(adapter_args)

    def update_queue(self, active_names: set[str]) -> None:
        new_queue: deque = deque()
        in_queue: set[str] = set()
        while self.source_queue:
            if (name := self.source_queue.popleft()) in active_names:
                new_queue.append(name)
                in_queue.add(name)
        for name in active_names:
            if name not in in_queue:
                new_queue.append(name)
        self.source_queue = new_queue

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        adapters = fetch_active_adapters()
        self.reconcile(adapters)
        if not self.sources:
            return []
        self.update_queue(set(self.sources))

        refs = {name: AdapterRef(name=name, slot=adapters[name].slot) for name in self.sources}
        reward_specs = {
            name: RewardSpec(
                rm_type=adapters[name].config.rm_type,
                custom_rm_path=adapters[name].config.custom_rm_path,
            )
            for name in self.sources
        }

        samples_per_adapter, remainder = divmod(num_samples, len(self.source_queue))
        all_samples: list[list[Sample]] = []
        to_deregister: list[str] = []

        for i in range(len(self.source_queue)):
            extra = int(i < remainder)
            samples_needed = samples_per_adapter + extra
            if samples_needed == 0:
                break
            name = self.source_queue.popleft()
            config = adapters[name].config
            self.source_queue.append(name)
            source = self.sources[name]
            adapter_samples = source.get_samples(samples_needed)
            ref = refs[name]
            reward_spec = reward_specs[name]
            for group in adapter_samples:
                for sample in group:
                    sample.adapter = ref
                    sample.reward_spec = reward_spec
                    sample.metadata = {**config.metadata, **sample.metadata}
            all_samples.extend(adapter_samples)

            default_num_row = (getattr(config, "num_epoch", 1) or 1) * len(source.dataset)
            num_row = config.num_row or default_num_row
            if source.sample_group_index >= num_row:
                logger.info(f"Adapter '{name}' reached num_row={num_row}, deregistering")
                to_deregister.append(name)

        for name in to_deregister:
            ray.get(get_multi_lora_controller().deregister_adapter.remote(name))

        return all_samples

    def add_samples(self, samples: list[list[Sample]]) -> None:
        """Recycle retried/aborted groups; drop groups for deregistered adapters."""
        adapters = fetch_active_adapters()
        self.reconcile(adapters)
        for group in samples:
            name = group[0].adapter.name if group and group[0].adapter else None
            if not name or name not in self.sources or name not in adapters:
                continue
            self.sources[name].add_samples([group])

    def save(self, rollout_id):
        for source in self.sources.values():
            source.save(rollout_id)

    def load(self, rollout_id=None):
        for source in self.sources.values():
            source.load(rollout_id)
