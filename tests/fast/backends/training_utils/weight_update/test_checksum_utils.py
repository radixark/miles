from typing import Any

import pytest
import torch

from miles.backends.training_utils.weight_update.checksum_utils import (
    compute_send_checksums,
    hash_tensor_sha256,
    verify_transfer_checksums,
)


def _rank_record(rank: int, checksums: dict[str, str], *, roles: tuple[str, ...] = ("tp",)) -> dict[str, Any]:
    return {"checksums": checksums, "parallelism_info": [{"role": role, "rank": rank} for role in roles]}


def _body(*records: dict[str, Any], success: bool = True) -> dict[str, Any]:
    return {"success": success, "ranks": list(records)}


def _verify(sent: dict[str, str], body: dict[str, Any], *, rank: int = 0) -> None:
    verify_transfer_checksums(sent_checksums=sent, engine_body=body, cell_id="cell-a", rank=rank)


class TestComputeSendChecksums:
    def test_each_name_is_hashed_from_its_own_buffer(self) -> None:
        """Swapping or reusing buffers across names would make a correct transfer look corrupt, or the reverse."""
        parameters = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0]), "unsent": torch.tensor([9.0])}

        checksums = compute_send_checksums(parameters, ["w", "b"])

        assert checksums == {"w": hash_tensor_sha256(parameters["w"]), "b": hash_tensor_sha256(parameters["b"])}
        assert checksums["w"] != checksums["b"]

    def test_the_hash_covers_the_bytes_currently_in_the_buffer(self) -> None:
        """The send buffer is reloaded per rank, so the hash must follow what is in it right now."""
        buffer = torch.tensor([1.0, 2.0])
        before = compute_send_checksums({"w": buffer}, ["w"])

        buffer[1] = 5.0

        assert compute_send_checksums({"w": buffer}, ["w"]) != before

    def test_a_duplicate_name_in_the_bucket_is_rejected(self) -> None:
        """A name written twice in one session means the bucket was assembled wrongly."""
        with pytest.raises(AssertionError, match="duplicate tensor names"):
            compute_send_checksums({"w": torch.zeros(2)}, ["w", "w"])

    def test_a_buffer_that_is_not_on_the_cpu_is_rejected(self) -> None:
        """Only the registered CPU buffers are what the transfer engine reads."""
        with pytest.raises(AssertionError, match="registered CPU send buffers"):
            compute_send_checksums({"w": torch.zeros(2, device="meta")}, ["w"])


class TestVerifyTransferChecksums:
    def test_identical_checksums_on_the_target_rank_pass(self) -> None:
        """The healthy transfer must stay accepted, or every failure below proves nothing."""
        _verify({"w": "h1", "b": "h2"}, _body(_rank_record(0, {"w": "h1", "b": "h2"})))

    def test_a_tensor_arriving_with_different_bytes_fails_and_is_named(self) -> None:
        """A silently corrupted write is exactly what the check exists to catch."""
        with pytest.raises(RuntimeError, match=r"1 tensors reached rollout cell cell-a rank 0 .*\['b'\]"):
            _verify({"w": "h1", "b": "h2"}, _body(_rank_record(0, {"w": "h1", "b": "bad"})))

    def test_a_sent_tensor_missing_on_the_receiver_fails(self) -> None:
        """A tensor the receiver did not report may never have landed."""
        with pytest.raises(RuntimeError, match=r"\['b'\]"):
            _verify({"w": "h1", "b": "h2"}, _body(_rank_record(0, {"w": "h1"})))

    def test_an_extra_tensor_reported_by_the_receiver_fails(self) -> None:
        """Checksums of tensors never sent mean the receiver answered for a different set of names."""
        with pytest.raises(RuntimeError, match=r"\['extra'\]"):
            _verify({"w": "h1"}, _body(_rank_record(0, {"w": "h1", "extra": "h9"})))

    def test_only_the_record_of_the_written_rank_is_compared(self) -> None:
        """Another rank of the same engine holds another shard and must not be compared against this write."""
        body = _body(_rank_record(0, {"w": "other-shard"}), _rank_record(1, {"w": "h1"}))

        _verify({"w": "h1"}, body, rank=1)
        with pytest.raises(RuntimeError, match="rank 0"):
            _verify({"w": "h1"}, body, rank=0)

    def test_a_failed_engine_body_is_rejected(self) -> None:
        """A receiver that could not compute checksums must not pass as verified."""
        with pytest.raises(AssertionError, match="engine reported failure"):
            _verify({"w": "h1"}, _body(_rank_record(0, {"w": "h1"}), success=False))

    @pytest.mark.parametrize("ranks", [[1], [0, 0]], ids=["missing", "duplicated"])
    def test_the_written_rank_must_appear_exactly_once(self, ranks: list[int]) -> None:
        """No record or two records for one rank leaves no single answer to compare."""
        body = _body(*[_rank_record(rank, {"w": "h1"}) for rank in ranks])

        with pytest.raises(AssertionError, match="expected one checksum record for GPU rank 0"):
            _verify({"w": "h1"}, body)

    def test_roles_disagreeing_on_the_gpu_rank_are_rejected(self) -> None:
        """A record whose roles name two GPU ranks cannot be attributed to one written rank."""
        record = {"checksums": {"w": "h1"}, "parallelism_info": [{"role": "tp", "rank": 0}, {"role": "ep", "rank": 1}]}

        with pytest.raises(AssertionError, match="expected one GPU rank across roles"):
            _verify({"w": "h1"}, _body(record))

    def test_roles_agreeing_on_the_gpu_rank_are_one_record(self) -> None:
        """Several roles naming the same GPU rank describe one record, not a conflict."""
        _verify({"w": "h1"}, _body(_rank_record(0, {"w": "h1"}, roles=("tp", "ep"))))
