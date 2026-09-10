import hashlib
from types import ModuleType

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.p2p_checksums import (
    P2PChecksumShard,
    checksum_transfer_tensors,
    merge_transfer_checksums,
)


class TestTransferTensorChecksums:
    def test_scalar_bfloat16_parameters_keep_their_raw_representation(self) -> None:
        """Scalar and bfloat16 parameters remain hashable without converting their values."""
        tensor = torch.tensor(-0.0, dtype=torch.bfloat16)
        assert checksum_transfer_tensors({"scale": tensor}, names=["scale"]) == {
            "scale": hashlib.sha256(b"\x00\x80").hexdigest()
        }

    def test_strided_send_buffers_use_the_exact_logical_bytes(self) -> None:
        """Hash transmitted bytes rather than the backing storage or a textual tensor representation."""
        tensor = torch.tensor([97, 0, 98, 0, 99], dtype=torch.uint8)[::2]
        checksums = checksum_transfer_tensors({"weight": tensor}, names=["weight"])
        assert checksums == {"weight": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
        tensor[0] = 100
        assert checksum_transfer_tensors({"weight": tensor}, names=["weight"]) != checksums

    def test_repeated_bucket_names_are_rejected(self) -> None:
        """A dictionary must not hide a duplicated parameter in the send stream."""
        with pytest.raises(AssertionError, match="repeats a parameter"):
            checksum_transfer_tensors({"w": torch.zeros(1)}, names=["w", "w"])


class TestMergeTransferChecksums:
    def test_pipeline_shards_merge_without_mixing_cells_or_engine_ranks(
        self, checksum_shards: list[P2PChecksumShard]
    ) -> None:
        """Each receiver gets the complete union of its own pipeline shards."""
        assert merge_transfer_checksums(checksum_shards[::-1], healthy_engine_ranks={"cell-a": 2, "cell-b": 1}) == {
            "cell-a": {"0": {"w": "a0-w", "b": "a0-b"}, "1": {"w": "a1-w", "b": "a1-b"}},
            "cell-b": {"0": {"w": "b0-w"}},
        }

    @pytest.mark.parametrize(
        "case, error",
        [
            ("missing_rank", "Missing P2P checksum sender"),
            ("missing_tensor", "Incomplete P2P tensor coverage"),
            ("extra_tensor", "Incomplete P2P tensor coverage"),
            ("duplicate_tensor", "Duplicate P2P tensor sender"),
            ("stale_session", "P2P receiver changed"),
            ("changed_parameters", "P2P receiver parameter sets disagree"),
            ("wrong_rank", "Unexpected P2P checksum rank"),
        ],
    )
    def test_incomplete_or_ambiguous_transfer_evidence_is_rejected(
        self, checksum_shards: list[P2PChecksumShard], case: str, error: str
    ) -> None:
        """Missing bytes, duplicate writers and receiver identity disagreement cannot authorize publication."""
        if case == "missing_rank":
            checksum_shards.pop(2)
        elif case == "missing_tensor":
            checksum_shards[1].tensors.clear()
        elif case == "extra_tensor":
            checksum_shards[1].tensors["extra"] = "extra-hash"
        elif case == "duplicate_tensor":
            checksum_shards[1].tensors["w"] = "a0-w"
        elif case == "stale_session":
            checksum_shards[1].session_id = "replacement-session"
        elif case == "changed_parameters":
            checksum_shards[1].expected_names = frozenset({"b"})
        elif case == "wrong_rank":
            checksum_shards[2].engine_rank = 2
        with pytest.raises(AssertionError, match=error):
            merge_transfer_checksums(checksum_shards, healthy_engine_ranks={"cell-a": 2, "cell-b": 1})

    def test_failed_cells_do_not_supply_or_require_publication_manifests(
        self, checksum_shards: list[P2PChecksumShard]
    ) -> None:
        """An incomplete failed target cannot prevent verification of the surviving target."""
        checksum_shards[0].tensors.clear()
        assert merge_transfer_checksums(checksum_shards, healthy_engine_ranks={"cell-b": 1}) == {
            "cell-b": {"0": {"w": "b0-w"}}
        }
        assert merge_transfer_checksums(checksum_shards, healthy_engine_ranks={}) == {}


class TestP2PChecksumWiring:
    @pytest.mark.parametrize("is_sender", [False, True])
    def test_rank_specific_send_buffers_reach_the_final_manifest_and_reset_next_update(
        self, checksum_protocol: tuple, p2p_protocol: ModuleType, monkeypatch: pytest.MonkeyPatch, is_sender: bool
    ) -> None:
        """Collectives include non-senders and each update hashes buffers before another rank overwrites them."""
        protocol, recorder = checksum_protocol
        protocol.begin_sync(weight_version=1, iter_buckets=lambda: iter(()))
        protocol.send_bucket([("hf.w", torch.zeros(1))])
        assert recorder.sent == [(0, b"a"), (1, b"b")]
        sender_shards = list(protocol._checksum_shards.values())
        protocol.is_sender = is_sender
        if not is_sender:
            protocol._checksum_shards = {}

        def gather(gathered: list, local: list, group: object) -> None:
            assert bool(local) == is_sender
            gathered[:] = [local, [] if is_sender else sender_shards]

        monkeypatch.setattr(p2p_protocol, "get_gloo_group", lambda: None)
        monkeypatch.setattr(p2p_protocol.dist, "get_world_size", lambda group: 2)
        monkeypatch.setattr(p2p_protocol.dist, "all_gather_object", gather)
        protocol.finalize(weight_version=1)
        assert protocol.expected_base_weight_checksums_by_cell == {
            "cell-a": {
                "0": {"w": hashlib.sha256(b"a").hexdigest()},
                "1": {"w": hashlib.sha256(b"b").hexdigest()},
            }
        }
        protocol.begin_sync(weight_version=2, iter_buckets=lambda: iter(()))
        assert protocol.expected_base_weight_checksums_by_cell is None
        assert all(not shard.tensors for shard in protocol._checksum_shards.values())

    def test_disabled_checksums_do_not_add_a_collective(
        self, checksum_protocol: tuple, p2p_protocol: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabling checksum capture leaves the training collective sequence unchanged."""
        protocol, _ = checksum_protocol
        protocol.args.save_inference_engine_weight_checksum = False

        def unexpected_collective(*args: object, **kwargs: object) -> None:
            raise AssertionError("Checksum collective ran while disabled")

        monkeypatch.setattr(p2p_protocol.dist, "all_gather_object", unexpected_collective)
        protocol.finalize(weight_version=1)
        assert protocol.expected_base_weight_checksums_by_cell is None
