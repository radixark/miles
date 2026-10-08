from typing import Any

import pytest
import torch

_WAIT_BOUND = 10.0
_STILL_BLOCKED_SECONDS = 0.2
# the target model's own MTP layer drafts, as sglang resolves NEXTN
_EAGLE_MTP = {"speculative_algorithm": "EAGLE", "speculative_draft_model_path": "/model"}


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
    def test_every_failed_write_is_named_and_fails_the_update(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A failed write leaves its rollout engine rank on the old weights, so the update must not succeed; the error
        names each failed rank, the last one included (`main` only logged those), and no other."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=3)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        p2p_sender.transfer_engine.failing_sessions = {api.session_id(1), api.session_id(2)}

        protocol.send_bucket(make_bucket("hf.w"))

        with pytest.raises(RuntimeError, match="2 of 3 p2p writes failed") as failure:
            protocol.after_base_weights()
        assert api.session_id(1) in str(failure.value) and api.session_id(2) in str(failure.value)
        assert api.session_id(0) not in str(failure.value)

    def test_a_write_still_running_at_the_timeout_fails_the_update(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A write that has not finished within --p2p-transfer-timeout fails the update instead of being forgotten."""
        protocol = p2p_sender.make_protocol(p2p_transfer_timeout=0.1)
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        p2p_sender.transfer_engine.hold(api.session_id(0))

        protocol.send_bucket(make_bucket("hf.w"))

        with pytest.raises(RuntimeError, match="still running after 0.1s"):
            protocol.after_base_weights()


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


class TestDraftRunner:
    def test_mtp_weights_reach_the_draft_and_weights_it_shares_go_once_through_the_target(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """p2p used to write the target by the draft's weight table and never write the draft. A weight the draft
        shares with the target, sent through the draft as well, would carry bytes its loader may not fill."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1, speculative_args=_EAGLE_MTP)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)

        protocol.send_bucket(make_bucket("hf.w", "hf.q", "hf.k", "hf.mtp"))
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "w"): [1.0, 2.0, 3.0, 4.0],
            api.target_address(0, "qk"): [5.0, 6.0, 7.0, 8.0],
        }
        assert p2p_sender.transfer_engine.payload_of(api.session_id(0, "draft")) == {
            api.target_address(0, "mtp"): [9.0, 10.0, 11.0, 12.0]
        }

    @pytest.mark.parametrize(
        "selector, speculative_args",
        [("target", _EAGLE_MTP), ("all", {"speculative_algorithm": "NGRAM"})],
        ids=["trainer_without_mtp", "no_draft_model"],
    )
    def test_only_the_target_is_written_when_there_is_no_draft_to_update(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, selector: str, speculative_args: dict
    ) -> None:
        """Without MTP layers in the trainer the draft keeps its own weights, and an engine without a draft model
        publishes no draft to query."""
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
        ],
        ids=["multi_layer_eagle", "draft_checkpoint"],
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
        """One sender writes the same runners on every engine it serves, so a draft would go unwritten on one engine
        or be queried on another that has none."""
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
