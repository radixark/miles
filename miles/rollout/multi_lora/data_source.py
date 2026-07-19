"""Round-robin per-adapter data source. Deregistration is step-based and
lives in the controller (``mark_batch_trained``); every adapter gets a
``num_step`` at registration, explicit or derived from ``num_epoch``."""

import copy
import logging
from argparse import Namespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import ray

from miles.ray.multi_lora_controller import get_multi_lora_controller
from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.adapter_config import AdapterRun
from miles.utils.types import AdapterRef, RewardSpec, Sample

logger = logging.getLogger(__name__)

MAX_RECONCILE_WORKERS = 16


def fetch_snapshot() -> dict:
    return ray.get(get_multi_lora_controller().snapshot.remote())


def sampleable(snapshot: dict) -> dict[str, AdapterRun]:
    return {**snapshot["active"], **snapshot["retiring"]}


class MultiLoRAAsyncDataSource(DataSource):
    def __init__(self, args: Namespace):
        self.args = args
        self.sources: dict[str, RolloutDataSource] = {}
        self.source_queue: deque = deque()

    def reconcile(self, adapters: dict[str, AdapterRun]) -> None:
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
                # Post-filter dataset length; the controller derives num_step
                # from num_epoch for adapters that didn't set it.
                ray.get(get_multi_lora_controller().resolve_num_step.remote(name, len(source.dataset)))
        self.update_queue(set(adapters))

    def create_source(self, adapter: AdapterRun) -> RolloutDataSource:
        config = adapter.config
        adapter_args = copy.copy(self.args)
        adapter_args.prompt_data = config.data
        adapter_args.input_key = config.input_key or self.args.input_key
        adapter_args.label_key = config.label_key or self.args.label_key
        adapter_args.metadata_key = config.metadata_key or self.args.metadata_key
        adapter_args.save = config.save or self.args.save
        adapter_args.load = config.save or self.args.load
        adapter_args.n_samples_per_prompt = config.n_samples_per_prompt or self.args.n_samples_per_prompt
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

    def get_samples(self, num_samples: int = 1) -> list[list[Sample]]:
        """Return the next prompt group, round-robined across adapters.

        One rotation of the queue: pull one group from the first adapter that
        yields, stamp it, and return. Empty list when no adapter can produce.
        """
        assert num_samples == 1, "the async producer dispatches one prompt group at a time"
        snapshot = fetch_snapshot()
        adapters = sampleable(snapshot)
        self.reconcile(adapters)
        self.update_queue(set(self.sources))

        for _ in range(len(self.source_queue)):
            name = self.source_queue.popleft()
            self.source_queue.append(name)
            source = self.sources[name]
            groups = source.get_samples(1)
            if not groups:
                continue

            adapter = adapters[name]
            config = adapter.config
            ref = AdapterRef(name=name, slot=adapter.slot)
            reward_spec = RewardSpec(rm_type=config.rm_type, custom_rm_path=config.custom_rm_path)
            for sample in groups[0]:
                sample.adapter = ref
                sample.reward_spec = reward_spec
                sample.metadata = {**config.metadata, **sample.metadata}

            return groups

        return []

    def add_samples(self, samples: list[list[Sample]]) -> None:
        """Recycle retried/aborted groups; drop groups for deregistered adapters."""
        adapters = sampleable(fetch_snapshot())
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

    def close(self) -> None:
        from miles.rollout.multi_lora.async_rollout import AsyncMultiLoRAWorker

        AsyncMultiLoRAWorker.stop_global()
