from typing import Any

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.utils.loader_probe import HfNameMapping

_WAIT_BOUND = 10.0
_STILL_BLOCKED_SECONDS = 0.2
# the target model's own MTP layer drafts, as sglang resolves NEXTN
_EAGLE_MTP = {"speculative_algorithm": "EAGLE", "speculative_draft_model_path": "/model"}


class TestConnectFiltersPeersWithoutMetadata:
    def test_a_cell_whose_metadata_query_fails_is_left_out_of_the_writes(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Planning a write to a cell with no session would crash the connect of every healthy cell."""
        protocol = p2p_sender.make_protocol()
        broken = make_rollout_api("cell-a", gpu_count=1, unreachable=True)
        healthy = make_rollout_api("cell-b", gpu_count=1)

        p2p_sender.connect(protocol, [broken, healthy])
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert protocol.is_sender
        assert protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert p2p_sender.transfer_engine.written_sessions() == [healthy.session_id(0)]

    def test_no_reachable_peer_leaves_the_rank_a_non_sender(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A rank left with no target must neither build the RDMA engine nor stage any weight."""
        protocol = p2p_sender.make_protocol()

        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1, unreachable=True)])
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(make_bucket("hf.q"))
        protocol.after_base_weights()

        assert protocol.is_sender is False
        assert p2p_sender.first_passes == 1
        assert p2p_sender.transfer_engines_created == 0
        assert p2p_sender.replicas_created == []
        assert p2p_sender.log == []

    def test_a_rank_the_plan_assigns_nothing_never_queries_the_engines(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """Querying engines this rank never writes to only spends the metadata deadline of every other rank."""
        p2p_sender.data_replica_rank = 1
        p2p_sender.data_replica_size = 2
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1)

        p2p_sender.connect(protocol, [api])

        assert protocol.is_sender is False
        assert api.calls == []


class TestSendBucket:
    def test_a_fused_parameter_is_loaded_and_written_only_once_every_shard_arrived(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Loading one shard of a fused parameter would write a half-filled parameter to the engine."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)

        protocol.send_bucket(make_bucket("hf.q"))
        log_after_first_shard = list(p2p_sender.log)
        protocol.send_bucket(make_bucket("hf.k"))
        protocol.after_base_weights()

        assert log_after_first_shard == []
        assert p2p_sender.log == [("load", 0, ("hf.q", "hf.k")), ("write", api.session_id(0))]
        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "qk"): [5.0, 6.0, 7.0, 8.0]
        }

    def test_a_parameter_still_missing_a_shard_fails_the_end_of_the_base_weights(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A shard that never arrives would otherwise leave the engine serving a stale parameter silently."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        p2p_sender.begin_sync(protocol, weight_version=1)

        protocol.send_bucket(make_bucket("hf.q"))

        with pytest.raises(AssertionError, match=r"\('qk',\) lacks \['hf.k'\]"):
            protocol.after_base_weights()

    def test_a_failed_write_gives_up_its_cell_and_the_other_cell_still_gets_every_rank(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """One broken cell must not abort the update of the cells that share its rollout engine ranks."""
        protocol = p2p_sender.make_protocol()
        broken, healthy = make_rollout_api("cell-a", gpu_count=2), make_rollout_api("cell-b", gpu_count=2)
        p2p_sender.connect(protocol, [broken, healthy])
        p2p_sender.begin_sync(protocol, weight_version=1)
        p2p_sender.transfer_engine.failing_sessions.add(broken.session_id(0))

        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert not protocol.cell_updaters_of_cell_id["cell-b"].is_errored
        assert sorted(p2p_sender.transfer_engine.written_sessions()) == [
            broken.session_id(0),
            healthy.session_id(0),
            healthy.session_id(1),
        ]


class TestTransferBuffers:
    def test_a_transfer_buffer_is_not_loaded_again_while_a_write_still_reads_it(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """With two buffers the third rank loads into the first rank's buffer; doing it while the first rank's write
        still reads that buffer would send the third rank's bytes to the first."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=3)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        first_write = p2p_sender.transfer_engine.hold(api.session_id(0))

        call = p2p_sender.call_in_thread(lambda: protocol.send_bucket(make_bucket("hf.w")))

        assert first_write.entered.wait(timeout=_WAIT_BOUND)
        assert p2p_sender.loaded_event(1).wait(timeout=_WAIT_BOUND)
        assert not p2p_sender.loaded_event(2).wait(timeout=_STILL_BLOCKED_SECONDS)
        first_write.release.set()
        call.join()
        protocol.after_base_weights()
        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "w"): [1.0, 2.0, 3.0, 4.0]
        }


class TestWriteThreads:
    def test_a_stuck_rollout_engine_does_not_hold_up_writes_to_another(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Each rollout engine has its own write thread, so a stuck engine delays only its own writes."""
        protocol = p2p_sender.make_protocol()
        stuck_api = make_rollout_api("cell-a", gpu_count=1)
        healthy_api = make_rollout_api("cell-b", gpu_count=1)
        p2p_sender.connect(protocol, [stuck_api, healthy_api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        stuck_write = p2p_sender.transfer_engine.hold(stuck_api.session_id(0))
        healthy_write = p2p_sender.transfer_engine.hold(healthy_api.session_id(0))

        protocol.send_bucket(make_bucket("hf.w"))

        assert stuck_write.entered.wait(timeout=_WAIT_BOUND)
        assert healthy_write.entered.wait(timeout=_WAIT_BOUND), "the healthy engine's write waited for the stuck one"
        p2p_sender.transfer_engine.release_all()
        protocol.after_base_weights()
        assert sorted(p2p_sender.transfer_engine.written_sessions()) == sorted(
            [stuck_api.session_id(0), healthy_api.session_id(0)]
        )


class TestWriteCompletion:
    def test_the_first_failed_rank_marks_its_cell_and_skips_later_ranks(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A failed rank excludes its whole cell and prevents queued later ranks from writing."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=3)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        p2p_sender.transfer_engine.failing_sessions = {api.session_id(1), api.session_id(2)}

        protocol.send_bucket(make_bucket("hf.w"))

        protocol.after_base_weights()
        assert protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert p2p_sender.transfer_engine.written_sessions() == [api.session_id(0), api.session_id(1)]

    def test_a_failed_write_marks_the_cell_before_its_transfer_buffer_is_loaded_again(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Waiting for the end of the base weights would queue every later write behind the broken rank first."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=3)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        p2p_sender.transfer_engine.failing_sessions = {api.session_id(0)}

        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()
        assert protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert api.session_id(2) not in p2p_sender.transfer_engine.written_sessions()

    def test_a_write_still_running_at_the_timeout_marks_its_cell(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A timed-out write removes its cell from the update while retaining its registered buffer."""
        protocol = p2p_sender.make_protocol(p2p_transfer_timeout=0.1)
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        p2p_sender.transfer_engine.hold(api.session_id(0))

        protocol.send_bucket(make_bucket("hf.w"))

        protocol.after_base_weights()
        assert protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert len(p2p_sender.transfer_engine.registered) == 2


class TestConnect:
    def test_ranks_are_assigned_over_the_rollout_engines_and_placement_handed_over(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Assigning over the engines in args queried engines that were not there and never wrote others; a fixed
        placement made every PP stage write the whole model under Megatron Bridge."""
        protocol = p2p_sender.make_protocol()
        protocol.args.rollout_num_gpus_per_engine = 1
        protocol.args.rollout_num_gpus = 1
        resolved_placement = object()
        api = make_rollout_api("cell-a", gpu_count=2)

        p2p_sender.connect(protocol, [api], placement=resolved_placement)
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.assignment_inputs == [(resolved_placement, [2])]
        assert p2p_sender.transfer_engine.written_sessions() == [api.session_id(0), api.session_id(1)]

    def test_rollout_engines_holding_one_rank_in_different_layouts_are_rejected(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """One model replica serves all engines of a rank, so their layouts must match."""
        protocol = p2p_sender.make_protocol()
        trtllm_api = make_rollout_api("cell-a", gpu_count=1, moe_runner_backend="flashinfer_trtllm")
        triton_api = make_rollout_api("cell-b", gpu_count=1, moe_runner_backend="triton")

        with pytest.raises(AssertionError, match="different layouts"):
            p2p_sender.connect(protocol, [trtllm_api, triton_api])

    def test_a_rollout_engine_placing_experts_a_replica_cannot_reproduce_is_rejected(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """The engine places these experts by metadata the replica does not have, so p2p would write them into the
        wrong slots; each such setting must be named."""
        protocol = p2p_sender.make_protocol()
        expert_placement = {
            "ep_num_redundant_experts": 32,
            "init_expert_location": "/placement.json",
            "enable_eplb": True,
            "ep_join_mode": "join",
            "elastic_ep_initial_size": 4,
            "dwdp_size": 2,
            "kt_weight_path": "/kt",
        }
        api = make_rollout_api("cell-a", gpu_count=1, expert_placement=expert_placement)

        with pytest.raises(AssertionError, match=f"rollout engine 0 places experts by {', '.join(expert_placement)},"):
            p2p_sender.connect(protocol, [api])

    def test_a_replica_that_does_not_match_the_published_weights_is_rejected(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """Bytes loaded in a layout the rollout engine does not hold would land in the wrong places."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1, published_weight_numel=2)

        with pytest.raises(
            AssertionError, match="does not match the weights the target of rollout engine 0 rank 0 publishes"
        ):
            p2p_sender.connect(protocol, [api])

    def test_a_reconnect_keeps_the_replica_and_the_registered_transfer_buffers(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Rebuilding the replica or its memory at a reconnect left the writes reading memory Mooncake never
        registered."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        replaced_api = make_rollout_api("cell-a", gpu_count=1, generation=2)
        p2p_sender.connect(protocol, [replaced_api])
        p2p_sender.begin_sync(protocol, weight_version=2)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.written_sessions()[-1] == replaced_api.session_id(0)
        assert len(p2p_sender.replicas_created) == 1
        assert p2p_sender.transfer_engines_created == 1
        assert len(p2p_sender.transfer_engine.registered) == 2

    def test_a_reconnect_whose_replicas_join_params_into_a_larger_group_grows_the_transfer_buffers(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A param group is loaded whole into one buffer; buffers sized at the first sync would fail every update
        after a reconnect whose new replicas share an HF tensor between two params."""
        protocol = p2p_sender.make_protocol(update_weight_buffer_size=64)
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()
        first_registrations = set(p2p_sender.transfer_engine.registered)

        replaced_api = make_rollout_api("cell-a", gpu_count=1, generation=2, moe_runner_backend="triton")
        p2p_sender.connect(protocol, [replaced_api])
        p2p_sender.replicas_created[-1].map_hf_names = (
            lambda hf_tensor_specs: HfNameMapping.from_hf_names_by_param_name(
                {"w": frozenset({"hf.w", "hf.k"}), "qk": frozenset({"hf.q", "hf.k"})}
            )
        )
        p2p_sender.begin_sync(protocol, weight_version=2)
        protocol.send_bucket(make_bucket("hf.w", "hf.q", "hf.k"))
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.written_sessions()[-1] == replaced_api.session_id(0)
        assert len(p2p_sender.transfer_engine.registered) == 2
        assert not first_registrations & set(p2p_sender.transfer_engine.registered)


class TestDraftRunner:
    @pytest.mark.parametrize("draft_shares_target_weights", [True, False], ids=["shared", "own_copy"])
    def test_each_runner_writes_every_weight_its_loader_loads(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, draft_shares_target_weights: bool
    ) -> None:
        """A draft's own copy of a target weight was never written when each HF tensor went to the first runner
        loading it; a weight it shares gets the same bytes from both runners."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api(
            "cell-a",
            gpu_count=1,
            speculative_args=_EAGLE_MTP,
            draft_shares_target_weights=draft_shares_target_weights,
        )
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)

        protocol.send_bucket(make_bucket("hf.w", "hf.q", "hf.k", "hf.mtp"))
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "w"): [1.0, 2.0, 3.0, 4.0],
            api.target_address(0, "qk"): [5.0, 6.0, 7.0, 8.0],
        }
        assert p2p_sender.transfer_engine.payload_of(api.session_id(0, "draft")) == {
            api.published_address(0, "w", "draft"): [1.0, 2.0, 3.0, 4.0],
            api.published_address(0, "mtp", "draft"): [9.0, 10.0, 11.0, 12.0],
        }

    @pytest.mark.parametrize(
        "selector, speculative_args",
        [("target", _EAGLE_MTP), ("all", {"speculative_algorithm": "NGRAM"})],
        ids=["trainer_without_mtp", "no_draft_model"],
    )
    def test_only_the_target_is_written_when_there_is_no_draft_to_update(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, selector: str, speculative_args: dict
    ) -> None:
        """Target-only updates must not query draft addresses."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1, speculative_args=speculative_args)
        p2p_sender.connect(protocol, [api], selector=selector)
        p2p_sender.begin_sync(protocol, weight_version=1)

        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.written_sessions() == [api.session_id(0)]
        assert not [call for call in api.calls if call.endswith(" draft")]

    @pytest.mark.parametrize(
        "speculative_args, reason",
        [
            ({**_EAGLE_MTP, "enable_multi_layer_eagle": True}, "multi-layer EAGLE"),
            ({"speculative_algorithm": "EAGLE3", "speculative_draft_model_path": "/eagle3-head"}, "own MTP layer"),
            ({**_EAGLE_MTP, "speculative_moe_runner_backend": "triton"}, "moe_runner_backend triton"),
            (
                {**_EAGLE_MTP, "moe_a2a_backend": "deepep", "speculative_moe_a2a_backend": "none"},
                "moe_a2a_backend none",
            ),
        ],
        ids=["multi_layer_eagle", "draft_checkpoint", "own_moe_runner_backend", "own_moe_a2a_backend"],
    )
    def test_a_draft_p2p_cannot_update_is_rejected_at_connect(
        self, p2p_sender: Any, make_rollout_api: Any, speculative_args: dict, reason: str
    ) -> None:
        """Left unwritten, the draft would fall behind the trained target and its proposals stop being accepted."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1, speculative_args=speculative_args)

        with pytest.raises(NotImplementedError, match=reason):
            p2p_sender.connect(protocol, [api])

    def test_rollout_engines_running_different_runners_are_rejected(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """Mixed roles would leave a draft unwritten or query one that does not exist."""
        protocol = p2p_sender.make_protocol()
        drafting_api = make_rollout_api("cell-a", gpu_count=1, speculative_args=_EAGLE_MTP)
        plain_api = make_rollout_api("cell-b", gpu_count=1)

        with pytest.raises(AssertionError, match="run different model runners"):
            p2p_sender.connect(protocol, [drafting_api, plain_api])


class TestHfNames:
    def test_the_trainer_names_are_collected_once_for_the_process(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """The first pass costs a sync's gather and convert; the names never change, so later syncs reuse them."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        for weight_version in (1, 2):
            p2p_sender.begin_sync(protocol, weight_version=weight_version)
            protocol.send_bucket(make_bucket("hf.w"))
            protocol.after_base_weights()

        assert p2p_sender.first_passes == 1

    def test_a_tensor_outside_the_first_pass_fails_the_send(self, p2p_sender: Any, make_rollout_api: Any) -> None:
        """No replica was mapped for it, so writing would guess where its bytes land."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        p2p_sender.begin_sync(protocol, weight_version=1)

        with pytest.raises(AssertionError, match="were not in the trainer's first pass"):
            protocol.send_bucket([("hf.unknown", torch.zeros(4))])
        assert p2p_sender.transfer_engine.writes == []

    def test_a_tensor_no_runner_loads_is_skipped(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """An MTP layer the engines do not draft with is one their own loaders ignore too."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)

        protocol.send_bucket(make_bucket("hf.mtp", "hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "w"): [1.0, 2.0, 3.0, 4.0]
        }


class TestWriteChecksums:
    @staticmethod
    def _send(p2p_sender: Any, protocol: Any, apis: list[Any], bucket: list[tuple[str, torch.Tensor]]) -> None:
        p2p_sender.connect(protocol, apis)
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(bucket)
        protocol.after_base_weights()

    def test_every_written_rank_is_verified_against_the_bytes_it_received(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Each rank gets its own shard, so the sent hash must be taken per rank from the reloaded buffer."""
        protocol = p2p_sender.make_protocol(check_weight_transfer_checksum=True)
        apis = [make_rollout_api("cell-a", gpu_count=2), make_rollout_api("cell-b", gpu_count=1)]

        self._send(p2p_sender, protocol, apis, make_bucket("hf.w"))

        assert [api.check_weights_calls for api in apis] == [
            [dict(action="raw_checksum", names=["w"], selector="target")] * 2,
            [dict(action="raw_checksum", names=["w"], selector="target")],
        ]
        assert not any(updater.is_errored for updater in protocol.cell_updaters_of_cell_id.values())

    def test_a_fused_parameter_is_verified_once_under_its_engine_name(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Only complete, transfer-ready engine tensors are written, so only they may be checked."""
        protocol = p2p_sender.make_protocol(check_weight_transfer_checksum=True)
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)

        protocol.send_bucket(make_bucket("hf.q"))
        protocol.send_bucket(make_bucket("hf.k"))
        protocol.after_base_weights()

        assert api.check_weights_calls == [dict(action="raw_checksum", names=["qk"], selector="target")]
        assert not protocol.cell_updaters_of_cell_id["cell-a"].is_errored

    def test_a_corrupted_receiver_fails_only_its_own_cell(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A single engine that received wrong bytes is dropped while its peers keep the update."""
        protocol = p2p_sender.make_protocol(check_weight_transfer_checksum=True)
        healthy, corrupted = make_rollout_api("cell-a", gpu_count=1), make_rollout_api("cell-b", gpu_count=1)
        corrupted.corrupted_ranks.add(0)

        self._send(p2p_sender, protocol, [healthy, corrupted], make_bucket("hf.w"))

        updaters = protocol.cell_updaters_of_cell_id
        assert (updaters["cell-a"].is_errored, updaters["cell-b"].is_errored) == (False, True)
        assert "cell-b rank 0" in str(updaters["cell-b"]._error)

    def test_a_mismatch_on_an_early_rank_stops_writing_that_cell(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """The earlier rank is drained before the next one, so its failure must keep later ranks of the cell unwritten."""
        protocol = p2p_sender.make_protocol(check_weight_transfer_checksum=True)
        api = make_rollout_api("cell-a", gpu_count=2)
        api.corrupted_ranks.add(0)

        self._send(p2p_sender, protocol, [api], make_bucket("hf.w"))

        assert protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert p2p_sender.transfer_engine.written_sessions() == [api.session_id(0)]

    def test_disabled_checksums_send_no_request_and_blame_nobody(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """With the flag off even a corrupting receiver is neither asked nor failed."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        api.corrupted_ranks.add(0)

        self._send(p2p_sender, protocol, [api], make_bucket("hf.w"))

        assert api.check_weights_calls == []
        assert not protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert sorted(p2p_sender.transfer_engine.written_sessions()) == [api.session_id(0), api.session_id(1)]


class TestReconnectionWithOutstandingWrites:
    @pytest.mark.parametrize("late_failure", [False, True])
    def test_a_new_generation_reuses_bounded_buffers_without_inheriting_old_failures(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, late_failure: bool
    ) -> None:
        """A timed-out reader can see reused bytes only in its removed cell and cannot blame its replacement."""
        protocol = p2p_sender.make_protocol(p2p_transfer_timeout=0.02)
        old_api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [old_api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        hold = p2p_sender.transfer_engine.hold(old_api.session_id(0))
        if late_failure:
            p2p_sender.transfer_engine.failing_sessions.add(old_api.session_id(0))
        protocol.send_bucket(make_bucket("hf.w"))
        assert hold.entered.wait(timeout=10)
        old_cell = protocol.cell_updaters_of_cell_id["cell-a"]
        (old_future,) = [
            write.future
            for writes in protocol._transfer_buffers._pending_writes_by_buffer_index
            for write in writes
            if write.cell_updater is old_cell
        ]
        protocol.after_base_weights()
        assert isinstance(old_cell._error, TimeoutError)
        registrations = list(p2p_sender.transfer_engine.registered)

        new_api = make_rollout_api("cell-a", gpu_count=1, generation=2)
        p2p_sender.connect(protocol, [new_api])
        p2p_sender.begin_sync(protocol, weight_version=2)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.send_bucket(make_bucket("hf.q", "hf.k"))
        protocol.after_base_weights()
        assert p2p_sender.transfer_engine.registered == registrations
        assert not protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert [session for session, _ in p2p_sender.transfer_engine.writes] == [new_api.session_id(0)] * 2

        hold.release.set()
        if late_failure:
            native_error = (
                "returned -1" if p2p_sender.p2p_transport == "mooncake" else "NIXL WRITE of 1 weights is ERR"
            )
            with pytest.raises(RuntimeError, match=native_error):
                old_future.result(timeout=10)
        else:
            old_future.result(timeout=10)
        assert p2p_sender.transfer_engine.payload_of(old_api.session_id(0)) == {
            old_api.target_address(0, "w"): [5.0, 6.0, 7.0, 8.0]
        }
        assert not protocol.cell_updaters_of_cell_id["cell-a"].is_errored
        assert isinstance(old_cell._error, TimeoutError)
        assert ("unregistered read", old_api.session_id(0)) not in p2p_sender.log

    @pytest.mark.parametrize("old_state", ["running", "native_error"])
    def test_buffer_growth_cannot_release_memory_a_failed_native_batch_may_read(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, old_state: str
    ) -> None:
        """Growing after a timeout or uncertain native error must retain the old allocations and fail explicitly."""
        protocol = p2p_sender.make_protocol(p2p_transfer_timeout=0.02, update_weight_buffer_size=64)
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        if old_state == "running":
            p2p_sender.transfer_engine.hold(api.session_id(0))
        else:
            p2p_sender.transfer_engine.failing_sessions.add(api.session_id(0))
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()
        registrations = list(p2p_sender.transfer_engine.registered)
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1, generation=2)])
        protocol.args.update_weight_buffer_size = 256

        with pytest.raises(RuntimeError, match="Cannot release"):
            p2p_sender.begin_sync(protocol, weight_version=2)

        assert protocol._transfer_buffers.buffer_nbytes == 64
        assert p2p_sender.transfer_engine.registered == registrations
        assert len(registrations) == 2


class TestDraftChecksumIsolation:
    @pytest.mark.parametrize("corrupted_role", ["target", "draft"])
    def test_corruption_in_either_runner_stops_both_roles_of_only_that_cell(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, corrupted_role: str
    ) -> None:
        """Draft and target checksum failures share cell fate while healthy cells keep both roles current."""
        protocol = p2p_sender.make_protocol(check_weight_transfer_checksum=True)
        bad = make_rollout_api("bad", gpu_count=1, speculative_args=_EAGLE_MTP, draft_shares_target_weights=False)
        good = make_rollout_api("good", gpu_count=1, speculative_args=_EAGLE_MTP, draft_shares_target_weights=False)
        bad.corrupted_roles.add(corrupted_role)
        p2p_sender.connect(protocol, [bad, good])
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()
        assert protocol.cell_updaters_of_cell_id["bad"].is_errored
        assert not protocol.cell_updaters_of_cell_id["good"].is_errored
        bad_writes = [write for write in p2p_sender.transfer_engine.writes if write[0].startswith("bad-")]

        protocol.send_bucket(make_bucket("hf.q", "hf.k", "hf.mtp"))
        protocol.after_base_weights()

        assert [write for write in p2p_sender.transfer_engine.writes if write[0].startswith("bad-")] == bad_writes
        assert good.check_weights_calls == [
            dict(action="raw_checksum", names=["w"], selector="target"),
            dict(action="raw_checksum", names=["w"], selector="draft"),
            dict(action="raw_checksum", names=["qk"], selector="target"),
            dict(action="raw_checksum", names=["mtp"], selector="draft"),
        ]
        assert not protocol.cell_updaters_of_cell_id["good"].is_errored


class TestWriteOrdering:
    def test_buckets_reach_an_engine_in_submission_order(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A later bucket must not overtake an earlier native write on the same engine."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        hold = p2p_sender.transfer_engine.hold(api.session_id(0))
        protocol.send_bucket(make_bucket("hf.w"))
        assert hold.entered.wait(timeout=10)
        protocol.send_bucket(make_bucket("hf.q", "hf.k"))
        assert p2p_sender.transfer_engine.writes == []

        hold.release.set()
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.writes == [
            (api.session_id(0), {api.target_address(0, "w"): [1.0, 2.0, 3.0, 4.0]}),
            (api.session_id(0), {api.target_address(0, "qk"): [5.0, 6.0, 7.0, 8.0]}),
        ]
