"""Refactor-equivalence capture for the operation identity / result plane
(codex-rollout-fullparameter-design-0810 §3.3): one selection's BatchPlan is
driven through the REAL production pipeline —

    batch_plan_to_metadata -> postprocess (DP pad) -> convert_samples_to_train_data
    -> tinker_loss_function -> _gather_logprobs -> commit_tinker_batch

— and every client-observable output is asserted against hand-computed
references: the exact loss value, the per-operation row-ordered logprobs, the
operation results (logprobs + metrics), and the dirty pins.

The batch-internal correlation keys (slot-keyed when this capture was written:
``tinker_loss_by_slot``/``operation_by_slot``; lane-keyed since §3.3 landed)
are deliberately forwarded key-agnostically between the pipeline stages,
exactly as ``miles/backends/training_utils/data.py`` forwards them: a refactor
that re-keys the correlation plane changes the key names but MUST reproduce
every assertion in this file unchanged — these are the invariants the tinker
SDK observes.

The plan's ``bound_slot`` values (5 and 1) deliberately differ from any real
registry slot: the result plane must correlate through the plan, never through
trainer residency.
"""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest
import torch

from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_inputs, make_parallel_state

from miles.backends.megatron_utils.tinker_backend.trainer import _gather_logprobs
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.losses import tinker_loss_function
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.ray.tinker_backend.backend import TinkerBackend
from miles.ray.tinker_backend.config import AdapterRunConfig
from miles.ray.tinker_backend.residency import ResidentBinding
from miles.rollout.tinker_backend.rollout_fn import batch_plan_to_metadata
from miles.utils.tinker_backend import BatchExecutionLease
from miles.utils.types import AdapterRef, Sample

VOCAB = 32

# One selection: A (CE, 2 rows) coalesced with B (importance sampling, 1 row).
PLAN = [
    dict(
        name="A",
        registration_id="r-A",
        bound_slot=5,
        operation_id="op-A",
        operation_kind="forward_backward",
        loss_spec={"loss_fn": "cross_entropy"},
        sample_count=2,
    ),
    dict(
        name="B",
        registration_id="r-B",
        bound_slot=1,
        operation_id="op-B",
        operation_kind="forward_backward",
        loss_spec={"loss_fn": "importance_sampling"},
        sample_count=1,
    ),
]

PROMPT_LENS = [4, 6, 5]
RESPONSE_LENS = [3, 5, 4]
LOSS_WEIGHTS = [[0.5, 0.0, 2.0], [1.0, 1.0, 0.0, -1.0, 0.25], [0.0, 0.0, 0.0, 0.0]]
ADVANTAGES = [[0.0, 0.0, 0.0], [0.0] * 5, [1.0, -1.0, 0.5, 2.0]]


def make_selection_samples(inputs) -> list[Sample]:
    """Three stamped rows exactly as the queue children emit them: row identity
    restarts per operation (A rows 0,1; B row 0), and the stamped slot is
    deliberately stale (9) — the plan is authoritative."""
    samples = []
    rows = [("A", 0), ("A", 1), ("B", 0)]
    for i, (name, row) in enumerate(rows):
        sample = Sample(
            tokens=inputs["unconcat_tokens"][i].tolist(),
            response_length=RESPONSE_LENS[i],
            loss_mask=[1] * RESPONSE_LENS[i],
            index=row,
            status=Sample.Status.COMPLETED,
            loss_weights=LOSS_WEIGHTS[i],
            advantages=ADVANTAGES[i],
            rollout_log_probs=inputs["rollout_log_probs"][i].tolist(),
        )
        sample.adapter = AdapterRef(name=name, registration_id=f"r-{name}", serving_version=1, slot=9)
        samples.append(sample)
    return samples


def make_pipeline(pad_to_dp_size: int | None = None):
    """Run the production conversion pipeline; returns (args, train_data,
    inputs, padded_row_count)."""
    make_parallel_state()
    loss_args = make_args(loss_type="custom_loss")
    inputs = make_inputs(
        seed=11,
        batch_size=3,
        prompt_lens=list(PROMPT_LENS),
        response_lens=list(RESPONSE_LENS),
        vocab_size=VOCAB,
        args=loss_args,
    )
    samples = make_selection_samples(inputs)
    if pad_to_dp_size is not None:
        convert_args = SimpleNamespace(
            multi_lora=True,
            use_dynamic_global_batch_size=True,
            disable_rollout_trim_samples=False,
            global_batch_size=8,
        )
        samples, post_metadata = postprocess_rollout_data(
            convert_args, samples, train_parallel_config={"dp_size": pad_to_dp_size}, pad_to_dp=True
        )
    lease = BatchExecutionLease(
        dispatch_id="lease-eq",
        bindings_by_operation=tuple(
            (
                entry["operation_id"],
                ResidentBinding((entry["name"], entry["registration_id"]), entry["bound_slot"]),
            )
            for entry in PLAN
        ),
    )
    metadata = batch_plan_to_metadata(PLAN, lease)
    convert_args = SimpleNamespace(use_dynamic_global_batch_size=False)
    train_data = convert_samples_to_train_data(
        convert_args,
        samples,
        metadata=metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    return loss_args, train_data, inputs, len(samples)


def loss_batch_from_train_data(args, train_data, inputs, n_rows: int) -> dict:
    """Build the loss micro-batch the way the training side does: tensorize
    the per-token channels and forward EVERY remaining tinker/adapter key
    verbatim (key-agnostic, mirroring miles/backends/training_utils/data.py's
    rollout-level forwarding) so a re-keyed correlation plane flows through
    without this test hard-coding today's key names."""
    unconcat = list(inputs["unconcat_tokens"])
    total_lens = list(inputs["total_lens"])
    if n_rows > len(unconcat):  # padded rows clone the donor (the last row)
        for _ in range(n_rows - len(unconcat)):
            unconcat.append(unconcat[-1])
            total_lens.append(total_lens[-1])
    batch = dict(
        unconcat_tokens=unconcat,
        total_lengths=total_lens,
        response_lengths=train_data["response_lengths"],
        loss_masks=[torch.tensor(m, dtype=torch.int32) for m in train_data["loss_masks"]],
        loss_weights=[torch.tensor(w, dtype=torch.float32) for w in train_data["loss_weights"]],
        advantages=[torch.tensor(a, dtype=torch.float32) for a in train_data["advantages"]],
        rollout_log_probs=[torch.tensor(r, dtype=torch.float32) for r in train_data["rollout_log_probs"]],
        tinker_logprob_collector={},
    )
    for key, value in train_data.items():
        batch.setdefault(key, value)
    return batch


def reference_log_probs(args, batch, logits):
    return get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"][: len(batch["total_lengths"])],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=None,
    )["log_probs"]


def expected_reference(args, batch, logits):
    """Hand-computed loss + per-row logprobs for the canonical selection:
    rows 0,1 are A's linear CE, row 2 is B's importance sampling; any padded
    row has all-zero mask/weights and contributes nothing."""
    lp = reference_log_probs(args, batch, logits)
    ce = sum(-(lp[i] * batch["loss_weights"][i] * batch["loss_masks"][i].float()).sum() for i in (0, 1))
    ratio = torch.exp(lp[2] - batch["rollout_log_probs"][2])
    is_loss = -(ratio * batch["advantages"][2] * batch["loss_masks"][2].float()).sum()
    return ce + is_loss, lp


class TestResultPlanePipeline:
    def test_loss_logprobs_and_commit_are_reproduced_field_by_field(self):
        args, train_data, inputs, n_rows = make_pipeline()
        assert n_rows == 3

        # -- conversion invariants (client-observable, key-agnostic) --
        assert train_data["batch_kind"] == "tinker"
        assert train_data["sample_indices"] == [0, 1, 0]  # row identity restarts per operation
        assert train_data["rewards"] == [0.0, 0.0, 0.0]  # tinker batches carry no rewards

        batch = loss_batch_from_train_data(args, train_data, inputs, n_rows)
        logits = inputs["policy_logits"].requires_grad_(True)
        loss, metrics = tinker_loss_function(args, batch, logits, sum_of_sample_mean=None)

        expected_loss, lp = expected_reference(args, batch, logits)
        assert torch.allclose(loss, expected_loss)
        assert torch.allclose(metrics["loss"], expected_loss)
        assert loss.requires_grad

        # -- result plane: rows group per OPERATION, in row order --
        rollout_data = {**train_data, "tinker_logprob_collector": batch["tinker_logprob_collector"]}
        logprobs_by_op = _gather_logprobs(rollout_data)
        assert set(logprobs_by_op) == {"op-A", "op-B"}
        assert logprobs_by_op["op-A"] == [pytest.approx(lp[0].tolist()), pytest.approx(lp[1].tolist())]
        assert logprobs_by_op["op-B"] == [pytest.approx(lp[2].tolist())]

        # -- commit: operations complete with row-ordered logprobs + metrics,
        #    and exactly the forward_backward registrations pin dirty --
        backend = self.make_backend_with_claimed_ops(logprobs_by_op)
        accumulated = [(name, backend.registry.find(name).registration_id) for name in ("A", "B")]
        backend.commit_tinker_batch(accumulated, ["op-A", "op-B"], logprobs_by_op)
        result_a = backend.operations.get("op-A")["result"]
        assert result_a["logprobs"] == logprobs_by_op["op-A"]
        expected_loss_sum = sum(
            -logprob * weight
            for row, weights in ((0, LOSS_WEIGHTS[0]), (1, LOSS_WEIGHTS[1]))
            for logprob, weight in zip(lp[row].tolist(), weights, strict=True)
        )
        assert result_a["metrics"]["loss:sum"] == pytest.approx(expected_loss_sum)
        assert result_a["metrics"]["unmasked_tokens:sum"] == 8.0
        result_b = backend.operations.get("op-B")["result"]
        assert result_b["logprobs"] == logprobs_by_op["op-B"]
        assert backend.registry.is_dirty("A") and backend.registry.is_dirty("B")

    def test_dp_padding_never_enters_the_result_plane(self):
        """7->8-style padding equivalence at 3->4: the padded clone of the last
        row carries zero mask/weights (no loss contribution) and the -1 row
        sentinel (excluded from every operation's logprobs)."""
        args, train_data, inputs, n_rows = make_pipeline(pad_to_dp_size=4)
        assert n_rows == 4
        assert train_data["sample_indices"] == [0, 1, 0, -1]
        assert train_data["loss_masks"][3] == [0, 0, 0, 0]
        assert train_data["loss_weights"][3] == [0.0, 0.0, 0.0, 0.0]
        assert train_data["advantages"][3] == [0.0, 0.0, 0.0, 0.0]

        batch = loss_batch_from_train_data(args, train_data, inputs, n_rows)
        # 4 rows need 4 logit streams: reuse the donor's logits for the clone.
        logits = torch.cat(
            [inputs["policy_logits"], inputs["policy_logits"][:, -inputs["total_lens"][-1] :]], dim=1
        ).requires_grad_(True)
        loss, _ = tinker_loss_function(args, batch, logits, sum_of_sample_mean=None)

        ref_batch = loss_batch_from_train_data(args, {**train_data}, inputs, n_rows)
        ref_batch["total_lengths"] = ref_batch["total_lengths"] + [inputs["total_lens"][-1]]
        expected_loss, lp = expected_reference(args, ref_batch, logits)
        assert torch.allclose(loss, expected_loss)  # the pad row moved nothing

        rollout_data = {**train_data, "tinker_logprob_collector": batch["tinker_logprob_collector"]}
        logprobs_by_op = _gather_logprobs(rollout_data)
        assert [len(rows) for rows in (logprobs_by_op["op-A"], logprobs_by_op["op-B"])] == [2, 1]

    @staticmethod
    def make_backend_with_claimed_ops(logprobs_by_op) -> TinkerBackend:
        backend_args = SimpleNamespace(
            multi_lora_n_adapters=4,
            save="/tmp/tinker-test-save",
            lora_rank=32,
            lora_alpha=64,
            hf_checkpoint="Qwen/Qwen3-0.6B",
        )
        backend = TinkerBackend(backend_args, "http://unused")
        payloads = {
            "op-A": {
                "samples": [
                    dict(
                        tokens=[1] * (PROMPT_LENS[i] + RESPONSE_LENS[i]),
                        response_length=RESPONSE_LENS[i],
                        loss_mask=[1] * RESPONSE_LENS[i],
                        loss_weights=LOSS_WEIGHTS[i],
                    )
                    for i in (0, 1)
                ],
                "loss": {"loss_fn": "cross_entropy"},
            },
            "op-B": {
                "samples": [
                    dict(
                        tokens=[1] * (PROMPT_LENS[2] + RESPONSE_LENS[2]),
                        response_length=RESPONSE_LENS[2],
                        loss_mask=[1] * RESPONSE_LENS[2],
                        advantages=ADVANTAGES[2],
                        rollout_log_probs=[-0.5] * RESPONSE_LENS[2],
                    )
                ],
                "loss": {"loss_fn": "importance_sampling"},
            },
        }
        for name, op_id in (("A", "op-A"), ("B", "op-B")):
            asyncio.run(backend.register(name, AdapterRunConfig()))
            backend.registry.mark_ready([name])
            rid = backend.registry.find(name).registration_id
            backend.enqueue_operation(name, op_id, 1, "forward_backward", payloads[op_id])
            assert backend.operations.claim_data_operation(name, rid)["operation_id"] == op_id
        return backend
