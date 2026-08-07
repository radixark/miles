import abc
import copy
import logging
import os
import random
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, NewType

import torch

from miles.utils.data import Dataset
from miles.utils.misc import load_function
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

SourceReservationId = NewType("SourceReservationId", str)


class SourceReservation(NamedTuple):
    """One source-owned prompt group attempt.

    Attributes:
        reservation_id: Stable logical identity across replay attempts.
        samples: Pristine samples for this attempt.
    """

    reservation_id: SourceReservationId
    samples: tuple[Sample, ...]


class _SourceReservationRecord(NamedTuple):
    group_index: int


class _OutstandingReservation(NamedTuple):
    record: _SourceReservationRecord
    attempt: SourceReservation


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    def get_buffer_length(self) -> int | None:
        """Pending-sample backlog, or None for sources without a buffer."""
        return None

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        """Reserve pristine prompt groups for ownership-aware rollout.

        Args:
            num_groups: Number of prompt groups to reserve.

        Returns:
            Reservations that require explicit settlement.

        Implementations must return exactly ``num_groups`` reservations with
        unique identities. If this method raises, it must not transfer source
        ownership.

        Raises:
            RuntimeError: If this data source does not support reservations.
        """
        raise RuntimeError(f"{self.__class__.__name__} does not support durable source reservations.")

    def acknowledge_reservations(self, reservations: Sequence[SourceReservation], *, rollout_id: int) -> None:
        """Record successful handoff of exact source reservation attempts.

        Args:
            reservations: Exact attempts to acknowledge.
            rollout_id: Training rollout that accepted the groups.

        Implementations must validate the complete batch before mutation. If
        this method raises, every input reservation must remain outstanding.

        Raises:
            RuntimeError: If this data source does not support reservations.
        """
        raise RuntimeError(f"{self.__class__.__name__} does not support durable source reservations.")

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        """Return exact source reservation attempts for pristine replay.

        Args:
            reservations: Exact attempts to requeue.

        Implementations must validate the complete batch before mutation. If
        this method raises, every input reservation must remain outstanding.

        Raises:
            RuntimeError: If this data source does not support reservations.
        """
        raise RuntimeError(f"{self.__class__.__name__} does not support durable source reservations.")


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}
        self._reservation_lock = threading.RLock()
        self._outstanding_reservations: dict[SourceReservationId, _OutstandingReservation] = {}
        self._replay_reservations: list[_SourceReservationRecord] = []
        self._durable_reservations_started = False
        self._permutation_epoch_id: int | None = None
        self._permutation: tuple[int, ...] = ()

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(
                args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
            )
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                # Bespoke processors (e.g. Inkling's) are not ProcessorMixin and cannot serialise.
                if hasattr(processor, "save_pretrained"):
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self._set_dataset_epoch(self.epoch_id)
        else:
            self.dataset = None

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        """Reserve prompt groups without settling source ownership.

        Args:
            num_groups: Number of groups to reserve.

        Returns:
            Replay attempts first, then newly admitted source groups.

        Raises:
            ValueError: If the group count or source dataset is invalid.
            RuntimeError: If durable reservations are unavailable.
        """
        self._require_durable_reservations()

        with self._reservation_lock:
            if not isinstance(num_groups, int) or isinstance(num_groups, bool) or num_groups < 0:
                raise ValueError(f"num_groups must be a nonnegative integer, got {num_groups!r}.")
            if num_groups == 0:
                return []
            if self.dataset is not None and len(self.dataset) == 0:
                raise ValueError("Cannot reserve samples from an empty rollout dataset.")

            replay_count = min(num_groups, len(self._replay_reservations))
            replay_records = self._replay_reservations[:replay_count]
            new_count = num_groups - replay_count
            new_records = [
                _SourceReservationRecord(group_index=group_index)
                for group_index in range(self.sample_group_index, self.sample_group_index + new_count)
            ]
            records = [*replay_records, *new_records]
            reservations = [self._materialize_reservation(record) for record in records]

            del self._replay_reservations[:replay_count]
            for record, reservation in zip(records, reservations, strict=True):
                self._outstanding_reservations[reservation.reservation_id] = _OutstandingReservation(
                    record=record,
                    attempt=reservation,
                )
            self._advance_source_frontier(new_records)
            if reservations:
                self._durable_reservations_started = True
            return reservations

    def acknowledge_reservations(self, reservations: Sequence[SourceReservation], *, rollout_id: int) -> None:
        """Acknowledge exact attempts after a training handoff succeeds.

        Args:
            reservations: Exact outstanding reservation attempts.
            rollout_id: Training rollout that accepted every parent group.

        Raises:
            ValueError: If the batch has duplicates or the rollout is invalid.
            RuntimeError: If any attempt is not currently outstanding.
        """
        self._require_durable_reservations()
        self._validate_rollout_id(rollout_id)
        with self._reservation_lock:
            outstanding = self._get_outstanding_reservations_locked(reservations)
            for owned in outstanding:
                reservation_id = owned.attempt.reservation_id
                del self._outstanding_reservations[reservation_id]

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        """Return exact outstanding attempts for pristine replay.

        Args:
            reservations: Exact attempts to reconstruct from source state.

        Raises:
            ValueError: If the batch contains duplicate identities.
            RuntimeError: If any attempt is not currently outstanding.
        """
        self._require_durable_reservations()
        with self._reservation_lock:
            outstanding = self._get_outstanding_reservations_locked(reservations)
            for owned in outstanding:
                del self._outstanding_reservations[owned.attempt.reservation_id]
                self._replay_reservations.append(owned.record)
            self._replay_reservations.sort(key=lambda record: record.group_index)

    def _get_outstanding_reservations_locked(
        self, reservations: Sequence[SourceReservation]
    ) -> list[_OutstandingReservation]:
        attempts = list(reservations)
        reservation_ids = [attempt.reservation_id for attempt in attempts]
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError(f"Reservation settlement contains duplicate identities: {reservation_ids}.")

        invalid = []
        outstanding = []
        for attempt in attempts:
            owned = self._outstanding_reservations.get(attempt.reservation_id)
            if owned is None or owned.attempt is not attempt:
                invalid.append(attempt.reservation_id)
            else:
                outstanding.append(owned)
        if invalid:
            raise RuntimeError(f"Source reservations are not the current outstanding attempts: {invalid}.")
        return outstanding

    def _require_durable_reservations(self) -> None:
        if not self.args.rollout_global_dataset:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support durable source reservations when rollout_global_dataset is disabled."
            )

    @staticmethod
    def _validate_rollout_id(rollout_id: int) -> None:
        if not isinstance(rollout_id, int) or isinstance(rollout_id, bool) or rollout_id < 0:
            raise ValueError(f"rollout_id must be a nonnegative integer, got {rollout_id!r}.")

    def _advance_source_frontier(self, records: list[_SourceReservationRecord]) -> None:
        if not records:
            return

        self.sample_group_index += len(records)
        self.sample_index = self.sample_group_index * self.args.n_samples_per_prompt
        if self.dataset is not None:
            epoch_id, epoch_offset = divmod(records[-1].group_index, len(self.dataset))
            self.epoch_id = epoch_id
            self.sample_offset = epoch_offset + 1
            if self.args.rollout_shuffle:
                self._set_dataset_epoch(epoch_id)

    def _materialize_reservation(self, record: _SourceReservationRecord) -> SourceReservation:
        assert self.dataset is not None
        epoch_id, epoch_offset = divmod(record.group_index, len(self.dataset))
        dataset_index = self._expected_dataset_index(epoch_id=epoch_id, epoch_offset=epoch_offset)
        prompt_sample = self.dataset.origin_samples[dataset_index]
        first_sample_index = record.group_index * self.args.n_samples_per_prompt
        samples = []
        for sample_index in range(first_sample_index, first_sample_index + self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.group_index = record.group_index
            sample.index = sample_index
            samples.append(sample)
        return SourceReservation(
            reservation_id=SourceReservationId(str(record.group_index)),
            samples=tuple(samples),
        )

    def _expected_dataset_index(self, *, epoch_id: int, epoch_offset: int) -> int:
        if not self.args.rollout_shuffle:
            return epoch_offset
        return self._dataset_permutation(epoch_id)[epoch_offset]

    def _dataset_permutation(self, epoch_id: int) -> tuple[int, ...]:
        assert self.dataset is not None
        if self._permutation_epoch_id == epoch_id:
            return self._permutation

        permutation = list(range(len(self.dataset)))
        random.Random(self.args.rollout_seed + epoch_id).shuffle(permutation)
        self._permutation_epoch_id = epoch_id
        self._permutation = tuple(permutation)
        return self._permutation

    def _set_dataset_epoch(self, epoch_id: int) -> None:
        assert self.dataset is not None
        permutation = self._dataset_permutation(epoch_id)
        self.dataset.samples = [self.dataset.origin_samples[index] for index in permutation]
        self.dataset.epoch_id = epoch_id

    def get_samples(self, num_samples):
        with self._reservation_lock:
            if self._durable_reservations_started:
                raise RuntimeError("Cannot use get_samples after durable source reservations have started.")
            # TODO further improve code
            if self.dataset is not None:
                if self.sample_offset + num_samples <= len(self.dataset):
                    prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                    self.sample_offset += num_samples
                else:
                    prompt_samples = self.dataset.samples[self.sample_offset :]
                    num_samples -= len(prompt_samples)
                    self.epoch_id += 1
                    if self.args.rollout_shuffle:
                        self._set_dataset_epoch(self.epoch_id)
                    prompt_samples += self.dataset.samples[:num_samples]
                    self.sample_offset = num_samples
            else:
                prompt_samples = [Sample() for _ in range(num_samples)]

            samples = []
            for prompt_sample in prompt_samples:
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt_sample)
                    sample.group_index = self.sample_group_index
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                self.sample_group_index += 1
                samples.append(group)
            return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        with self._reservation_lock:
            if self._durable_reservations_started:
                raise RuntimeError(
                    "Cannot save source state after durable source reservations have started "
                    "until reservation checkpointing is available."
                )
            state_dict = {
                "sample_offset": self.sample_offset,
                "epoch_id": self.epoch_id,
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "metadata": self.metadata,
            }
            path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        with self._reservation_lock:
            if self._durable_reservations_started:
                raise RuntimeError("Cannot load source state after durable source reservations have started.")
            logger.info(f"load metadata from {path}")
            logger.info(f"load metadata: {self.metadata}")
            state_dict = torch.load(path)
            self.sample_offset = state_dict.get("sample_offset", 0)
            self.epoch_id = state_dict.get("epoch_id", 0)
            self.sample_group_index = state_dict.get("sample_group_index", 0)
            self.sample_index = state_dict.get("sample_index", 0)
            self.metadata = state_dict.get("metadata", {})

            if self.args.rollout_shuffle:
                self._set_dataset_epoch(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        raise RuntimeError(
            f"{self.__class__.__name__} does not support durable source reservations because they would bypass its retry buffer."
        )

    def acknowledge_reservations(self, reservations: Sequence[SourceReservation], *, rollout_id: int) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} does not support durable source reservations because they would bypass its retry buffer."
        )

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} does not support durable source reservations because they would bypass its retry buffer."
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
