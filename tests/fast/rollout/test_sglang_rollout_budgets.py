"""CPU-only tests: no tokenizer, inference server, reward backend, or GPU required."""

import argparse
import asyncio
import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.rm_hub.math_dapo_strict_utils import compute_score
from miles.utils.types import Sample


@pytest.fixture
def isolated_modules(monkeypatch):
    """Load real production code with only the GPU-only dumper import stubbed.

    Use --noconftest on GPU-less ROCm hosts: the global fixtures import SGLang
    serving modules which inspect the local GPU at import time.
    """
    import miles.utils

    dumper = ModuleType("miles.utils.dumper_utils")
    dumper.configure_sglang = AsyncMock()
    monkeypatch.setitem(sys.modules, dumper.__name__, dumper)
    monkeypatch.setattr(miles.utils, "dumper_utils", dumper, raising=False)
    root = Path(__file__).resolve().parents[3]
    loaded = {}
    for name, path in (
        ("miles.rollout._budget_test_sglang_rollout", "miles/rollout/sglang_rollout.py"),
        ("miles.utils._budget_test_arguments", "miles/utils/arguments.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[path] = module
    return SimpleNamespace(
        rollout=loaded["miles/rollout/sglang_rollout.py"], arguments=loaded["miles/utils/arguments.py"]
    )


@pytest.fixture
def rollout(isolated_modules):
    return isolated_modules.rollout


@pytest.fixture
def env(monkeypatch, rollout):
    # Deliberately omit the new arguments: old test/plugin namespaces must work.
    args = argparse.Namespace(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        n_samples_per_prompt=2,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        partial_rollout=False,
        group_rm=False,
        reward_key=None,
        sglang_router_policy="round_robin",
        sglang_router_ip="own-rollout",
        sglang_router_port=30000,
        sglang_model_routers={"ref": ("other-model", 30001)},
        use_miles_router=False,
    )
    state = object.__new__(rollout.GenerateState)
    state.args = args
    state.sampling_params = {}
    state.reset()
    monkeypatch.setattr(rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(rollout, "_ROLLOUT_ABORT_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(rollout, "_ROLLOUT_CANCEL_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(rollout.sglang_router, "__version__", "0.3.0")
    monkeypatch.setattr(rollout.dumper_utils, "configure_sglang", AsyncMock())
    monkeypatch.setattr(rollout, "recompute_samples_rollout_logprobs_via_prefill", AsyncMock())
    monkeypatch.setattr(rollout, "call_agent_abort_hook", AsyncMock())
    get = AsyncMock(return_value={"workers": [{"url": "http://own-worker@0"}, {"url": "http://own-worker@1"}]})
    post = AsyncMock(return_value={})
    monkeypatch.setattr(rollout, "get", get)
    monkeypatch.setattr(rollout, "post", post)
    pbar = Mock()
    monkeypatch.setattr(rollout, "tqdm", Mock(return_value=pbar))

    tasks = []
    create_task = asyncio.create_task

    def track_task(coro, **kwargs):
        task = create_task(coro, **kwargs)
        tasks.append(task)
        return task

    monkeypatch.setattr(rollout.asyncio, "create_task", track_task)
    generated = []

    async def generate(_args, sample, _params, evaluation=False):
        generated.append(sample.index)
        sample.status = Sample.Status.COMPLETED
        sample.response = "answer"
        sample.reward = 1.0
        return sample

    monkeypatch.setattr(rollout, "generate_and_rm", generate)
    requested = []
    next_group = 0

    def data_source(count):
        nonlocal next_group
        requested.append(count)
        groups = [
            [Sample(index=i * args.n_samples_per_prompt + j, group_index=i) for j in range(args.n_samples_per_prompt)]
            for i in range(next_group, next_group + count)
        ]
        next_group += count
        return groups

    env = SimpleNamespace(
        args=args,
        state=state,
        tasks=tasks,
        data_source=data_source,
        generated=generated,
        generate=generate,
        requested=requested,
        get=get,
        post=post,
        pbar=pbar,
    )
    yield env
    assert all(task.done() for task in tasks), "rollout leaked pending tasks"
    assert not state.pendings


def reject_groups(monkeypatch, rollout, predicate=lambda _group: True):
    monkeypatch.setattr(
        rollout,
        "apply_preput_filters",
        lambda _args, _filter, group: FilterOutput(keep=not predicate(group), reason="test_reject"),
    )


async def test_legacy_namespace_completes_without_new_fields(env, rollout):
    output, aborted = await rollout.generate_rollout_async(env.args, 0, env.data_source)
    assert len(output.samples) == 2
    assert [sample.reward for group in output.samples for sample in group] == [1.0] * 4
    assert aborted == []
    assert env.requested == [2]
    assert env.state.submitted_candidate_groups == 0
    assert not env.state.aborted
    env.pbar.close.assert_called_once()
    env.get.assert_awaited_once_with("http://own-rollout:30000/workers")
    env.post.assert_awaited_once_with("http://own-worker/abort_request", {"abort_all": True})


async def test_candidate_budget_resets_for_each_outer_rollout(env, rollout):
    env.args.rollout_max_candidate_groups = 2
    env.args.rollout_timeout_seconds = 1.0
    for rollout_id in (0, 1):
        output, _ = await rollout.generate_rollout_async(env.args, rollout_id, env.data_source)
        assert len(output.samples) == 2
        assert env.state.submitted_candidate_groups == 0
    assert env.requested == [2, 2]
    assert len(env.generated) == 8


async def test_budget_clamps_oversampling_and_fails_explicitly(env, monkeypatch, rollout):
    env.args.over_sampling_batch_size = 32
    env.args.rollout_max_candidate_groups = 3
    reject_groups(monkeypatch, rollout)
    with pytest.raises(
        RuntimeError, match=r"insufficient valid groups \(0/2\).*candidate group budget exhausted"
    ) as exc:
        await rollout.generate_rollout_async(env.args, 7, env.data_source)
    assert "Rollout 7" in str(exc.value)
    assert "submitted_candidate_groups=3" in str(exc.value)
    assert env.requested == [3]
    assert len(env.generated) == 6


async def test_counts_actual_submissions_not_data_source_request_size(env, monkeypatch, rollout):
    env.args.over_sampling_batch_size = 8
    env.args.rollout_max_candidate_groups = 3
    reject_groups(monkeypatch, rollout)

    def short_source(count):
        return env.data_source(count)[:1]

    with pytest.raises(RuntimeError, match="submitted_candidate_groups=3"):
        await rollout.generate_rollout_async(env.args, 0, short_source)
    assert env.requested == [3, 2, 1]
    assert len(env.generated) == 6


async def test_unbounded_defaults_keep_replenishing_after_filter_drops(env, monkeypatch, rollout):
    reject_groups(monkeypatch, rollout, predicate=lambda group: group[0].group_index < 2)
    output, _ = await rollout.generate_rollout_async(env.args, 0, env.data_source)
    assert len(output.samples) == 2
    assert env.requested == [2, 2]
    assert output.metrics == {"rollout/dynamic_filter/drop_test_reject": 2}


async def test_exhausted_budget_allows_inflight_to_fill_target(env, monkeypatch, rollout):
    env.args.over_sampling_batch_size = 3
    env.args.rollout_max_candidate_groups = 3
    dropped = asyncio.Event()

    def filter_group(_args, _filter, group):
        if group[0].group_index == 0:
            dropped.set()
            return FilterOutput(keep=False, reason="test_reject")
        return FilterOutput(keep=True)

    async def delayed_generate(args, sample, params, evaluation=False):
        if sample.group_index:
            await dropped.wait()
        return await env.generate(args, sample, params, evaluation=evaluation)

    monkeypatch.setattr(rollout, "apply_preput_filters", filter_group)
    monkeypatch.setattr(rollout, "generate_and_rm", delayed_generate)
    output, _ = await rollout.generate_rollout_async(env.args, 0, env.data_source)
    assert len(output.samples) == 2
    assert [group[0].group_index for group in output.samples] == [1, 2]
    assert env.requested == [3]


async def test_budget_smaller_than_target_waits_then_fails(env, rollout):
    env.args.rollout_max_candidate_groups = 1
    with pytest.raises(RuntimeError, match=r"insufficient valid groups \(1/2\).*budget exhausted"):
        await rollout.generate_rollout_async(env.args, 0, env.data_source)
    assert env.requested == [1]
    assert len(env.generated) == 2


async def test_deadline_includes_waiting_and_cancels_all_tasks(env, monkeypatch, rollout):
    env.args.rollout_timeout_seconds = 0.02
    cancelled = []

    async def hung_generate(_args, sample, _params, evaluation=False):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(sample.index)

    monkeypatch.setattr(rollout, "generate_and_rm", hung_generate)
    with pytest.raises(RuntimeError, match=r"insufficient valid groups \(0/2\).*sampling deadline exceeded") as exc:
        await asyncio.wait_for(rollout.generate_rollout_async(env.args, 0, env.data_source), timeout=1.0)
    assert "submitted_candidate_groups=2" in str(exc.value)
    assert sorted(cancelled) == [0, 1, 2, 3]
    env.post.assert_awaited_once()
    env.pbar.close.assert_called_once()


async def test_deadline_includes_async_configuration(env, monkeypatch, rollout):
    env.args.rollout_timeout_seconds = 0.02

    async def hung_config(_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(rollout.dumper_utils, "configure_sglang", hung_config)
    with pytest.raises(RuntimeError, match="sampling deadline exceeded"):
        await rollout.generate_rollout_async(env.args, 0, env.data_source)
    assert env.requested == []


async def test_expired_deadline_never_submits_more_groups(env, rollout):
    # Runtime guard also protects direct callers bypassing argument validation.
    env.args.rollout_timeout_seconds = 0.0
    with pytest.raises(RuntimeError, match="sampling deadline exceeded"):
        await rollout.generate_rollout_async(env.args, 0, env.data_source)
    assert env.generated == []
    assert env.requested == []


async def test_failed_rollout_resets_budget_and_deadline_for_retry(env, monkeypatch, rollout):
    env.args.rollout_max_candidate_groups = 2
    env.args.rollout_timeout_seconds = 0.02

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(rollout, "generate_and_rm", hang)
    with pytest.raises(RuntimeError, match="sampling deadline exceeded"):
        await rollout.generate_rollout_async(env.args, 0, env.data_source)
    monkeypatch.setattr(rollout, "generate_and_rm", env.generate)
    output, _ = await rollout.generate_rollout_async(env.args, 1, env.data_source)
    assert len(output.samples) == 2
    assert env.requested == [2, 2]


@pytest.mark.parametrize("failure_site", ["get", "post", "hook", "drain"])
async def test_hung_abort_is_bounded_without_losing_original_error(env, monkeypatch, failure_site, rollout):
    original = ValueError("original generation failure")
    started = asyncio.Event()
    cancelled = []
    count = 0

    async def fail_one_sample(_args, sample, _params, evaluation=False):
        nonlocal count
        count += 1
        if count == 4:
            started.set()
        try:
            await started.wait()
            if sample.index == 0:
                raise original
            await asyncio.Event().wait()
        finally:
            cancelled.append(sample.index)

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(rollout, "generate_and_rm", fail_one_sample)
    if failure_site != "drain":
        monkeypatch.setattr(rollout, "call_agent_abort_hook" if failure_site == "hook" else failure_site, hang)
    with pytest.raises(ValueError) as exc:
        await asyncio.wait_for(rollout.generate_rollout_async(env.args, 0, env.data_source), timeout=1.0)
    assert exc.value is original
    assert sorted(cancelled) == [0, 1, 2, 3]


async def test_cleanup_error_does_not_mask_generation_error(env, monkeypatch, rollout):
    original = ValueError("generate exploded")

    async def fail(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(rollout, "generate_and_rm", fail)
    env.get.side_effect = RuntimeError("abort discovery exploded")
    with pytest.raises(ValueError) as exc:
        await rollout.generate_rollout_async(env.args, 0, env.data_source)
    assert exc.value is original


async def test_external_cancellation_reclaims_tasks_and_preserves_cancel(env, monkeypatch, rollout):
    started = asyncio.Event()

    async def hang(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(rollout, "generate_and_rm", hang)
    unrelated = asyncio.create_task(asyncio.Event().wait())
    task = asyncio.create_task(rollout.generate_rollout_async(env.args, 0, env.data_source))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert not unrelated.done()
    finally:
        unrelated.cancel()
        await asyncio.gather(unrelated, return_exceptions=True)


async def test_empty_data_source_fails_without_busy_loop(env, rollout):
    with pytest.raises(RuntimeError, match="data source returned no groups"):
        await rollout.generate_rollout_async(env.args, 0, lambda _count: [])
    assert env.generated == []


async def test_success_with_oversampling_cancels_unused_tasks(env, monkeypatch, rollout):
    env.args.over_sampling_batch_size = 3
    env.args.rollout_max_candidate_groups = 3

    async def hang_extra(args, sample, params, evaluation=False):
        if sample.group_index == 2:
            await asyncio.Event().wait()
        return await env.generate(args, sample, params, evaluation=evaluation)

    monkeypatch.setattr(rollout, "generate_and_rm", hang_extra)
    output, _ = await asyncio.wait_for(rollout.generate_rollout_async(env.args, 0, env.data_source), timeout=1.0)
    assert len(output.samples) == 2


async def test_successful_partial_rollout_still_collects_aborted_groups(env, monkeypatch, rollout):
    env.args.partial_rollout = True
    env.args.over_sampling_batch_size = 3
    env.args.rollout_max_candidate_groups = 3
    aborted = asyncio.Event()

    async def abort_post(*_args, **_kwargs):
        aborted.set()
        return {}

    async def partial_generate(args, sample, params, evaluation=False):
        if sample.group_index == 2:
            await aborted.wait()
            sample.status = Sample.Status.ABORTED
            sample.response = "partial"
            return sample
        return await env.generate(args, sample, params, evaluation=evaluation)

    monkeypatch.setattr(rollout, "post", abort_post)
    monkeypatch.setattr(rollout, "generate_and_rm", partial_generate)
    output, partials = await rollout.generate_rollout_async(env.args, 8, env.data_source)
    assert len(output.samples) == 2
    assert len(partials) == 1
    assert all(sample.metadata["start_rollout_id"] == 8 for sample in partials[0])


def configure_smoke_sampling(args):
    args.n_samples_per_prompt = 4
    args.over_sampling_batch_size = 8
    args.rollout_max_candidate_groups = 32
    args.rollout_timeout_seconds = 1800
    args.reward_key = "score"
    args.advantage_estimator = "grpo"
    args.rewards_normalization = True
    args.grpo_std_normalization = True
    args.normalize_advantages = False
    args.dynamic_sampling_filter_path = (
        "miles.rollout.filter_hub.truncated_response_filters.mask_truncated_and_require_completed_reward_diversity"
    )


@pytest.mark.parametrize("oversampling,initial_requests", [(2, 8), (8, 32)])
async def test_smoke_initial_fanout_uses_real_fill_loop(env, monkeypatch, rollout, oversampling, initial_requests):
    configure_smoke_sampling(env.args)
    env.args.over_sampling_batch_size = oversampling
    env.args.rollout_timeout_seconds = 0.04
    active = peak = 0

    async def stall(*_args, **_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    monkeypatch.setattr(rollout, "generate_and_rm", stall)
    with pytest.raises(RuntimeError, match="sampling deadline exceeded"):
        await asyncio.wait_for(rollout.generate_rollout_async(env.args, 2, env.data_source), timeout=1)
    assert env.requested == [oversampling]
    assert peak == initial_requests
    assert active == 0


async def test_smoke_refill_bounds_32_groups_and_36_inflight_requests(env, monkeypatch, rollout):
    configure_smoke_sampling(env.args)
    total = active = peak = 0
    release_straggler = asyncio.Event()
    submissions = []
    submit = env.state.submit_generate_tasks

    def track_submission(groups):
        submit(groups)
        submissions.append((env.state.submitted_candidate_groups, len(env.state.pendings)))

    async def all_wrong(_args, sample, _params, evaluation=False):
        nonlocal total, active, peak
        total += 1
        active += 1
        peak = max(peak, active)
        if total == 32 * 4:
            release_straggler.set()
        try:
            if sample.group_index == 0:
                await release_straggler.wait()
            else:
                await asyncio.sleep(0)  # yield so all submitted sample tasks start
            sample.status = Sample.Status.COMPLETED
            sample.tokens = [1, 2, 3]
            sample.response_length = 2
            sample.response = "Answer: 0"
            sample.label = "1"
            sample.reward = compute_score(sample.response, sample.label)
            return sample
        finally:
            active -= 1

    monkeypatch.setattr(env.state, "submit_generate_tasks", track_submission)
    monkeypatch.setattr(rollout, "generate_and_rm", all_wrong)
    with pytest.raises(RuntimeError, match="candidate group budget exhausted"):
        await asyncio.wait_for(rollout.generate_rollout_async(env.args, 3, env.data_source), timeout=2)
    assert env.requested == [8, 8, 8, 8]
    assert max(count for count, _ in submissions) == 32
    assert max(pending for _, pending in submissions) == 9
    assert (total, peak, active) == (128, 36, 0)


@pytest.fixture
def strict_candidates(env, monkeypatch, rollout):
    configure_smoke_sampling(env.args)
    observed = []

    async def generate(_args, sample, _params, evaluation=False):
        sample.status = Sample.Status.COMPLETED
        sample.tokens = [1, 2, 3]
        sample.response_length = 2
        sample.label = "1"
        position = sample.index % 4
        if sample.group_index == 0:  # all correct
            sample.response = "Answer: 1"
        elif sample.group_index == 1:  # all wrong
            sample.response = "Answer: 0"
        elif sample.group_index == 2:  # invalid format is NOT a mathematical error
            sample.response = "I think it is 1" if position == 0 else "Answer: 1"
        elif sample.group_index == 3:  # truncation cannot supply completed diversity
            sample.response = "Answer: 1"
            if position:
                sample.status = Sample.Status.TRUNCATED
        else:  # accepted only with completed correct + incorrect evidence
            sample.response = f"Answer: {position % 2}"
        sample.reward = compute_score(
            sample.response, sample.label, is_complete=sample.status == Sample.Status.COMPLETED
        )
        observed.append(sample)
        return sample

    monkeypatch.setattr(rollout, "generate_and_rm", generate)
    return SimpleNamespace(generate=generate, observed=observed)


@pytest.mark.parametrize("dump", [False, True])
@pytest.mark.parametrize("failure", ["budget", "deadline"])
async def test_failure_evidence_and_summary_keep_rejected_candidates(
    env, monkeypatch, rollout, strict_candidates, tmp_path, caplog, dump, failure
):
    template = str(tmp_path / "rollout_data" / "{rollout_id}.pt")
    env.args.save_debug_rollout_data = template if dump else None
    if failure == "budget":
        env.args.rollout_max_candidate_groups = 4
        message = "candidate group budget exhausted"
    else:
        env.args.rollout_timeout_seconds = 0.06
        message = "sampling deadline exceeded"

        async def stall_after_four(args, sample, params, evaluation=False):
            if sample.group_index >= 4:
                await asyncio.Event().wait()
            return await strict_candidates.generate(args, sample, params, evaluation=evaluation)

        monkeypatch.setattr(rollout, "generate_and_rm", stall_after_four)
    with pytest.raises(RuntimeError, match=message):
        await asyncio.wait_for(rollout.generate_rollout_async(env.args, 7, env.data_source), timeout=1)
    log = next(record.message for record in caplog.records if "candidate summary=" in record.message)
    for text in (
        "'complete_groups': 4",
        "'fully_completed_groups': 3",
        "'invalid_samples': 4",
        "'format_invalid_samples': 1",
        "'truncated_samples': 3",
        "'rewards':",
        "'zero_reward_std': 2",
        "'no_completed_reward_diversity': 2",
        "'valid_groups': 0",
    ):
        assert text in log
    path = tmp_path / "rollout_data" / "candidates" / "7.failure.pt"
    assert f"evidence_file={path if dump else None}" in caplog.text
    assert not (tmp_path / "rollout_data" / "7.pt").exists()
    if not dump:
        assert list(tmp_path.rglob("*")) == []
        return
    payload = torch.load(path, weights_only=False)
    assert payload["outcome"] == "failure" and payload["diagnostic_only"]
    assert message in payload["error"]
    assert payload["summary"]["rewards"] == {"1.0": 8, "-1.0": 8}
    assert len(payload["candidate_groups"]) == 4
    assert "samples" not in payload  # incompatible with training/replay schema
    for candidate in payload["candidate_groups"]:
        assert candidate["filter_keep"] is False
        assert candidate["selected_for_batch"] is False
        for row in candidate["samples"]:
            assert row["reward"] == compute_score(
                row["response"], row["label"], is_complete=row["status"] == "completed"
            )
            assert "rollout_routed_experts" not in row and "tokens" not in row
    load_args = SimpleNamespace(load_debug_rollout_data=str(path), load_debug_rollout_data_subsample=None)
    with pytest.raises(ValueError, match="not training/replay input"):
        load_debug_rollout_data(load_args, 7)
    with pytest.raises(ValueError, match="not training/replay input"):
        RolloutDataInjectionUtil.load(SimpleNamespace(ci_inject_rollout_data_path=str(path)), 7)


async def test_success_evidence_contains_rejected_and_accepted_without_changing_training_dump(
    env, rollout, strict_candidates, tmp_path, monkeypatch
):
    env.args.rollout_max_candidate_groups = 6
    env.args.save_debug_rollout_data = str(tmp_path / "{rollout_id}.pt")
    output, _ = await rollout.generate_rollout_async(env.args, 8, env.data_source)
    assert [group[0].group_index for group in output.samples] == [4, 5]
    payload = torch.load(tmp_path / "candidates" / "8.success.pt", weights_only=False)
    assert len(payload["candidate_groups"]) == 6
    assert sum(group["selected_for_batch"] for group in payload["candidate_groups"]) == 2
    assert sum(group["filter_keep"] for group in payload["candidate_groups"]) == 2
    assert payload["summary"]["filter_reason_counts"] == {
        "zero_reward_std": 2,
        "no_completed_reward_diversity": 2,
    }
    assert output.metrics == {
        "rollout/dynamic_filter/drop_zero_reward_std": 2,
        "rollout/dynamic_filter/drop_no_completed_reward_diversity": 2,
    }
    assert all(not sample.remove_sample for sample in strict_candidates.observed)
    assert all(
        sample.reward
        == compute_score(sample.response, sample.label, is_complete=sample.status == Sample.Status.COMPLETED)
        for sample in strict_candidates.observed
    )
    # The existing outer success saver still saves ONLY selected training inputs.
    env.args.save_debug_trajectory_data = None
    monkeypatch.setattr("miles.ray.rollout.debug_data.save_dashboard_columns", lambda *_: None)
    save_debug_rollout_data(env.args, [s for group in output.samples for s in group], 8, evaluation=False)
    train = torch.load(tmp_path / "8.pt", weights_only=False)
    assert len(train["samples"]) == 8
    assert {sample["group_index"] for sample in train["samples"]} == {4, 5}


async def test_dump_io_error_preserves_sampling_failure_and_summary(env, rollout, monkeypatch, caplog):
    env.args.rollout_max_candidate_groups = 2
    reject_groups(monkeypatch, rollout)
    monkeypatch.setattr(rollout, "save_rollout_candidate_evidence", Mock(side_effect=OSError("disk full")))
    with pytest.raises(RuntimeError, match="candidate group budget exhausted"):
        await rollout.generate_rollout_async(env.args, 9, env.data_source)
    assert "candidate evidence write failed" in caplog.text
    assert "'complete_groups': 2" in caplog.text and "'test_reject': 2" in caplog.text
    assert "evidence_file=None" in caplog.text


async def test_completed_unconsumed_siblings_are_evidence_not_training(env, rollout, monkeypatch, tmp_path):
    env.args.save_debug_rollout_data = str(tmp_path / "fixed.pt")  # no template placeholder
    original = ValueError("filter exploded")
    monkeypatch.setattr(rollout, "apply_preput_filters", Mock(side_effect=original))
    with pytest.raises(ValueError) as exc:
        await rollout.generate_rollout_async(env.args, 10, env.data_source)
    assert exc.value is original
    payload = torch.load(tmp_path / "candidates" / "fixed.failure.pt", weights_only=False)
    assert payload["summary"]["complete_groups"] == 2
    assert payload["summary"]["unfiltered_groups"] == 2
    assert len(payload["candidate_groups"]) == 2
    assert not (tmp_path / "fixed.pt").exists()


async def test_success_evidence_includes_valid_surplus_not_selected_for_training(
    env, rollout, strict_candidates, tmp_path
):
    env.args.save_debug_rollout_data = str(tmp_path / "{rollout_id}.pt")
    output, _ = await rollout.generate_rollout_async(env.args, 11, env.data_source)
    payload = torch.load(tmp_path / "candidates" / "11.success.pt", weights_only=False)
    assert len(payload["candidate_groups"]) == 8
    assert sum(group["filter_keep"] for group in payload["candidate_groups"]) == 4
    assert sum(group["selected_for_batch"] for group in payload["candidate_groups"]) == 2
    assert len(output.samples) == 2


async def test_preput_filter_reason_counts_are_in_failure_evidence(env, rollout, monkeypatch, tmp_path):
    env.args.rollout_max_candidate_groups = 2
    env.args.save_debug_rollout_data = str(tmp_path / "{rollout_id}.pt")

    async def invalid(args, sample, params, evaluation=False):
        await env.generate(args, sample, params, evaluation=evaluation)
        if sample.group_index == 0:
            sample.status = Sample.Status.ABORTED
        else:
            sample.reward = None
        return sample

    monkeypatch.setattr(rollout, "generate_and_rm", invalid)
    with pytest.raises(RuntimeError, match="candidate group budget exhausted"):
        await rollout.generate_rollout_async(env.args, 12, env.data_source)
    payload = torch.load(tmp_path / "candidates" / "12.failure.pt", weights_only=False)
    assert payload["summary"]["filter_reason_counts"] == {
        "group_has_aborted": 1,
        "group_has_missing_reward": 1,
    }


@pytest.mark.parametrize("dump", [False, True])
async def test_failure_evidence_does_not_retain_candidates_across_rollouts(env, rollout, monkeypatch, tmp_path, dump):
    env.args.rollout_max_candidate_groups = 2
    env.args.save_debug_rollout_data = str(tmp_path / "{rollout_id}.pt") if dump else None
    reject_groups(monkeypatch, rollout)
    refs = []

    async def tracked(*args, **kwargs):
        sample = await env.generate(*args, **kwargs)
        refs.append(weakref.ref(sample))
        return sample

    monkeypatch.setattr(rollout, "generate_and_rm", tracked)
    for rollout_id in range(3):
        try:
            await rollout.generate_rollout_async(env.args, rollout_id, env.data_source)
        except RuntimeError:
            pass
        else:
            pytest.fail("all rejected candidates must fail")
        # The fixture deliberately tracks tasks; release that test-only retention.
        env.tasks.clear()
        await asyncio.sleep(0)
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert env.state.submitted_candidate_groups == 0
        assert not env.state.pendings


def test_budget_cli_defaults_and_types(monkeypatch, isolated_modules):
    monkeypatch.setattr("sys.argv", ["pytest"])
    parser = isolated_modules.arguments.get_miles_extra_args_provider()(argparse.ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.rollout_max_candidate_groups is None
    assert defaults.rollout_timeout_seconds is None
    args = parser.parse_args(["--rollout-max-candidate-groups", "32", "--rollout-timeout-seconds", "900.5"])
    assert args.rollout_max_candidate_groups == 32
    assert args.rollout_timeout_seconds == 900.5
    isolated_modules.arguments.validate_rollout_sampling_budgets(args)


def test_existing_dump_details_resolves_rollout_save_template(monkeypatch, isolated_modules, tmp_path):
    monkeypatch.setattr("sys.argv", ["pytest"])
    parser = isolated_modules.arguments.get_miles_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--dump-details",
            str(tmp_path),
            "--num-rollout",
            "1",
            "--rollout-batch-size",
            "2",
            "--rollout-function-path",
            "miles.rollout.sglang_rollout.generate_rollout",
        ]
    )
    isolated_modules.arguments.miles_validate_args(args)
    assert args.save_debug_rollout_data == f"{tmp_path}/rollout_data/{{rollout_id}}.pt"
    assert args.load_debug_rollout_data is None
    assert args.rollout_all_samples_process_path is None


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_reject_invalid_candidate_budget(value, isolated_modules):
    with pytest.raises(ValueError, match="positive integer"):
        isolated_modules.arguments.validate_rollout_sampling_budgets(
            SimpleNamespace(rollout_max_candidate_groups=value)
        )


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_reject_invalid_deadline(value, isolated_modules):
    with pytest.raises(ValueError, match="finite and positive"):
        isolated_modules.arguments.validate_rollout_sampling_budgets(SimpleNamespace(rollout_timeout_seconds=value))


def test_argument_validation_accepts_legacy_namespace(isolated_modules):
    isolated_modules.arguments.validate_rollout_sampling_budgets(SimpleNamespace())


@pytest.mark.parametrize("legacy", [True, False])
def test_budget_flags_cannot_silently_use_unsupported_rollout(isolated_modules, monkeypatch, legacy):
    monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
    args = SimpleNamespace(
        partial_rollout=False,
        fully_async=False,
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout" if legacy else None,
        eval_function_path=None,
        eval_num_gpus=0,
        rollout_max_candidate_groups=32,
    )
    if legacy:
        isolated_modules.arguments._resolve_rollout_functions(args)
    else:
        with pytest.raises(ValueError, match="other rollout implementations do not enforce these budgets"):
            isolated_modules.arguments._resolve_rollout_functions(args)


async def test_deadline_reports_nonempty_but_insufficient_batch(env, monkeypatch, rollout):
    env.args.rollout_timeout_seconds = 0.02

    async def hang_second(args, sample, params, evaluation=False):
        if sample.group_index == 1:
            await asyncio.Event().wait()
        return await env.generate(args, sample, params, evaluation=evaluation)

    monkeypatch.setattr(rollout, "generate_and_rm", hang_second)
    with pytest.raises(RuntimeError, match=r"insufficient valid groups \(1/2\).*sampling deadline exceeded"):
        await rollout.generate_rollout_async(env.args, 0, env.data_source)


async def test_overproducing_source_cannot_exceed_submission_budget(env, monkeypatch, rollout):
    env.args.rollout_max_candidate_groups = 2
    reject_groups(monkeypatch, rollout)
    with pytest.raises(RuntimeError, match="submitted_candidate_groups=2"):
        await rollout.generate_rollout_async(env.args, 0, lambda count: env.data_source(count + 5))
    assert len(env.generated) == 4


async def test_partial_group_task_submission_failure_reclaims_children(env, rollout):
    env.args.sglang_enable_deterministic_inference = True
    env.state.group_sampling_seeds = [0]  # The second sample fails after the first task was created.
    with pytest.raises(IndexError):
        await rollout.generate_rollout_async(env.args, 0, env.data_source)


async def test_simultaneous_failures_are_all_retrieved(env, monkeypatch, rollout):
    errors = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))

    async def fail(*_args, **_kwargs):
        raise ValueError("simultaneous generation failures")

    monkeypatch.setattr(rollout, "generate_and_rm", fail)
    try:
        with pytest.raises(ValueError, match="simultaneous generation failures"):
            await rollout.generate_rollout_async(env.args, 0, env.data_source)
        assert all(not getattr(task, "_log_traceback", False) for task in env.tasks)
        assert errors == []
    finally:
        loop.set_exception_handler(previous_handler)


async def test_abort_task_ignoring_cancel_cannot_block_forever(env, monkeypatch, rollout):
    # asyncio cannot kill a coroutine that ignores cancellation; bound the wait,
    # retain an exception-consumption callback, then release the fake for teardown.
    release = asyncio.Event()
    ignored_cancel = asyncio.Event()
    monkeypatch.setattr(rollout, "_ROLLOUT_CANCEL_TIMEOUT_SECONDS", 0.02)

    async def stubborn_get(_url):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            ignored_cancel.set()
            await release.wait()
            raise RuntimeError("late abort failure") from None

    monkeypatch.setattr(rollout, "get", stubborn_get)
    reject_groups(monkeypatch, rollout)
    env.args.rollout_max_candidate_groups = 2
    try:
        with pytest.raises(RuntimeError, match="candidate group budget exhausted"):
            await asyncio.wait_for(rollout.generate_rollout_async(env.args, 0, env.data_source), timeout=1.0)
        assert ignored_cancel.is_set()
    finally:
        release.set()
        await asyncio.gather(*env.tasks, return_exceptions=True)
