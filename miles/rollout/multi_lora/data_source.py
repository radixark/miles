"""No-op manager-level data source: the real per-adapter sources live inside
``MultiLoRARolloutFn``. The snapshot helpers are shared with scoped aborts."""

import logging
from argparse import Namespace

import ray

from miles.ray.multi_lora.controller import get_multi_lora_controller
from miles.rollout.data_source import DataSource
from miles.utils.adapter_config import AdapterRun
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def fetch_snapshot() -> dict:
    return ray.get(get_multi_lora_controller().snapshot.remote())


def sampleable(snapshot: dict) -> dict[str, AdapterRun]:
    return {**snapshot["active"], **snapshot["retiring"]}


class MultiLoRANullDataSource(DataSource):
    """No-op DataSource: the wrapper owns one real source per registration, so
    there is nothing to sample, save, or load at the manager level."""

    def __init__(self, args: Namespace):
        self.args = args

    def get_samples(self, num_samples: int = 1) -> list[list[Sample]]:
        return []

    def add_samples(self, samples: list[list[Sample]]) -> None:
        pass

    def save(self, rollout_id) -> None:
        pass

    def load(self, rollout_id=None) -> None:
        pass
