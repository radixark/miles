from copy import deepcopy
from dataclasses import fields
from itertools import pairwise
from typing import Any

from miles.rollout.generate_utils.rollout_topk_logprobs import merge_rollout_topk_logprobs_field
from miles.rollout.generate_utils.sampling_mask import merge_sampling_masks
from miles.utils.output_store.plan import combine_plans
from miles.utils.types import Sample

_OPD_STUDENT_TOP_LOGPROBS_KEY = "opd_student_top_logprobs"


_REPLAY_FIELDS = (
    "rollout_routed_experts",
    "rollout_indexer_topk",
    "rollout_sampling_mask",
)


def merge_samples(samples: list[Sample], tokenizer) -> Sample:
    turns = samples[:1]
    for sample in samples[1:]:
        # Only a COMPLETED turn can be extended by a later turn; if an
        # intermediate turn truncated, the trajectory ends there.
        # TODO (shi.dong): figure out how in-turn truncation should be handled.
        if turns[-1].status != Sample.Status.COMPLETED:
            break
        # Replay metadata must be complete through the merged prefix.
        if _introduces_replay_gap(turns[-1], sample):
            break
        turns.append(sample)
    return turns[0] if len(turns) == 1 else _merge_turns(turns, tokenizer)


def _introduces_replay_gap(a: Sample, b: Sample) -> bool:
    return any(
        getattr(a, field) is not None and getattr(b, field) is None for field in _REPLAY_FIELDS
    )  # config-access-exempt: attribute selected at runtime from field


def _merge_turns(turns: list[Sample], tokenizer) -> Sample:
    """Merge the samples of a trajectory's turns, each extending the turn before it.

    Neither input is modified. Each per-token field is built once from every turn, with each
    observation between them: merging turn by turn copied the growing trajectory on every turn,
    quadratic in its number of turns. The result shares the inputs' values it keeps unchanged,
    such as the last turn's tokens and prompt.
    """
    first, last = turns[0], turns[-1]
    observations = [
        later.tokens[len(earlier.tokens) : len(later.tokens) - later.response_length]
        for earlier, later in pairwise(turns)
    ]
    obs_lens = [len(later.tokens) - len(earlier.tokens) - later.response_length for earlier, later in pairwise(turns)]
    # TODO: is this acceptable?
    obs_texts = [tokenizer.decode(observation) for observation in observations]

    def _joined(per_turn: list[list], fill) -> list:
        """Each turn's values, with ``fill(obs_len)`` for each observation between them."""
        joined = list(per_turn[0])
        for obs_len, values in zip(obs_lens, per_turn[1:], strict=True):
            joined += fill(obs_len)
            joined += values
        return joined

    def _per_token(field: str, *, default, observation) -> list:
        """A per-token field over the trajectory: ``default`` where a turn has none, ``observation`` between turns."""
        return _joined(
            [
                value if (value := getattr(turn, field)) is not None else [default] * turn.response_length
                for turn in turns
            ],  # config-access-exempt: iterate the explicit sample completion field list
            lambda n: [observation] * n,
        )

    def _merge_optional_per_token(field):
        # Optional OPD per-token lists (teacher_log_probs, opd_reverse_kl): merge like
        # rollout_log_probs when present (zeros over the injected observation span), else keep None.
        if all(getattr(turn, field) is None for turn in turns):  # config-access-exempt: explicit OPD fields
            return None
        return _per_token(field, default=0.0, observation=0.0)

    def _merge_metadata():
        shared, top_logprobs, lifecycle, messages = None, [], [], None
        for index, turn in enumerate(turns):
            metadata, turn_top_logprobs = _pop_opd_student_top_logprobs(turn.metadata)
            metadata, turn_lifecycle = _pop_lifecycle(metadata)
            metadata, turn_messages = _pop_messages(metadata)
            if index == 0:
                shared = metadata
            assert metadata == shared, f"metadata mismatch: a.metadata={first.metadata}, b.metadata={turn.metadata}"
            top_logprobs.append(turn_top_logprobs)
            lifecycle += turn_lifecycle
            messages = turn_messages or messages

        merged_metadata = deepcopy(shared)
        if any(value is not None for value in top_logprobs):
            assert all(
                value is not None for value in top_logprobs
            ), f"{_OPD_STUDENT_TOP_LOGPROBS_KEY} must be present on every turn when merging top-k OPD metadata"
            for turn, value in zip(turns, top_logprobs, strict=True):
                assert len(value) == turn.response_length, (
                    f"{_OPD_STUDENT_TOP_LOGPROBS_KEY} length mismatch: {_OPD_STUDENT_TOP_LOGPROBS_KEY} has length "
                    f"{len(value)}, response_length={turn.response_length}"
                )
            merged_metadata = merged_metadata if merged_metadata is not None else {}
            merged_metadata[_OPD_STUDENT_TOP_LOGPROBS_KEY] = _joined(top_logprobs, lambda n: [[] for _ in range(n)])
        if lifecycle:
            merged_metadata = merged_metadata if merged_metadata is not None else {}
            merged_metadata["lifecycle"] = lifecycle
        if messages is not None:
            merged_metadata = merged_metadata if merged_metadata is not None else {}
            merged_metadata["messages"] = messages
        return merged_metadata

    def _merge_equal_value(field):
        x = getattr(first, field)  # config-access-exempt: attribute selected at runtime from field
        for turn in turns[1:]:
            y = getattr(turn, field)  # config-access-exempt: attribute selected at runtime from field
            assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    for turn in turns:
        turn.validate()
    store_plan = first.rollout_store_plan
    for (earlier, later), obs_len in zip(pairwise(turns), obs_lens, strict=True):
        try:
            assert _startswith(short=earlier.prompt, long=later.prompt), "b.prompt must start with a.prompt"
            assert _startswith(short=earlier.tokens, long=later.tokens), "b.tokens must start with a.tokens"
            assert obs_len > 0, f"obs_len must be > 0, got {obs_len}"
            store_plan = combine_plans(
                store_plan,
                later.rollout_store_plan,
                lambda x, y, obs_len=obs_len: x.merge_turn(y, observation_tokens=obs_len),
            )
            if earlier.rollout_routed_experts is not None:
                assert (
                    later.rollout_routed_experts is not None
                ), "cannot merge: a has rollout_routed_experts but b does not"
                assert earlier.rollout_routed_experts.shape[0] <= later.rollout_routed_experts.shape[0]
            if earlier.rollout_indexer_topk is not None:
                assert (
                    later.rollout_indexer_topk is not None
                ), "cannot merge: a has rollout_indexer_topk but b does not"
                assert earlier.rollout_indexer_topk.shape[0] <= later.rollout_indexer_topk.shape[0]
            assert earlier.status == Sample.Status.COMPLETED, f"a.status must be COMPLETED, got {earlier.status}"
        except AssertionError as e:
            if hasattr(
                e, "add_note"
            ):  # config-access-exempt: exception notes are unavailable on older Python versions
                e.add_note(f"a={earlier} b={later}")
            raise

    spec_info, prefix_cache_info = first.spec_info, first.prefix_cache_info
    for turn in turns[1:]:
        spec_info = _merge_spec_info(spec_info, turn.spec_info)
        prefix_cache_info = _merge_prefix_cache_info(prefix_cache_info, turn.prefix_cache_info)

    merged = _create_with_all_fields(
        Sample,
        group_index=_merge_equal_value("group_index"),
        index=_merge_equal_value("index"),
        lineage=_merge_equal_value("lineage"),
        rollout_id=_merge_equal_value("rollout_id"),
        prompt=last.prompt,
        tokens=last.tokens,
        multimodal_inputs=_merge_equal_value("multimodal_inputs"),
        multimodal_train_inputs=_merge_equal_value("multimodal_train_inputs"),
        response=first.response
        + "".join(text + turn.response for text, turn in zip(obs_texts, turns[1:], strict=True)),
        response_length=sum(turn.response_length for turn in turns) + sum(obs_lens),
        label=_merge_equal_value("label"),
        reward=_merge_equal_value("reward"),
        loss_mask=_per_token("loss_mask", default=1, observation=0),
        weight_versions=[call for turn in turns for call in turn.weight_versions],
        kv_cache_namespace=_merge_equal_value("kv_cache_namespace"),
        rollout_log_probs=_per_token("rollout_log_probs", default=0.0, observation=0.0),
        rollout_topk_token_ids=merge_rollout_topk_logprobs_field(turns, "rollout_topk_token_ids", obs_lens),
        rollout_topk_log_probs=merge_rollout_topk_logprobs_field(turns, "rollout_topk_log_probs", obs_lens),
        rollout_sampling_mask=merge_sampling_masks(turns, observations),
        teacher_log_probs=_merge_optional_per_token("teacher_log_probs"),
        opd_reverse_kl=_merge_optional_per_token("opd_reverse_kl"),
        rollout_routed_experts=last.rollout_routed_experts,
        rollout_indexer_topk=last.rollout_indexer_topk,
        rollout_store_plan=store_plan,
        remove_sample=_merge_equal_value("remove_sample"),
        status=last.status,
        metadata=_merge_metadata(),
        generate_function_path=_merge_equal_value("generate_function_path"),
        max_response_len=_merge_equal_value("max_response_len"),
        agent_function_path=_merge_equal_value("agent_function_path"),
        max_seq_len=_merge_equal_value("max_seq_len"),
        train_metadata=_merge_equal_value("train_metadata"),
        reward_spec=_merge_equal_value("reward_spec"),
        routing_key=_merge_equal_value("routing_key"),
        trainer_model_id=_merge_equal_value("trainer_model_id"),
        non_generation_time=_merge_equal_value("non_generation_time"),
        spec_info=spec_info,
        prefix_cache_info=prefix_cache_info,
    )
    merged.validate()
    return merged


def _pop_opd_student_top_logprobs(metadata):
    if metadata is None:
        return None, None
    metadata = dict(metadata)
    top_logprobs = metadata.pop(_OPD_STUDENT_TOP_LOGPROBS_KEY, None)
    return metadata, top_logprobs


def _pop_lifecycle(metadata):
    if not metadata or "lifecycle" not in metadata:
        return metadata, []
    value = metadata["lifecycle"]
    rest = {k: v for k, v in metadata.items() if k != "lifecycle"}
    return rest, value if isinstance(value, list) else [value]


def _pop_messages(metadata):
    if not metadata or "messages" not in metadata:
        return metadata, None
    return {k: v for k, v in metadata.items() if k != "messages"}, metadata["messages"]


def _merge_spec_info(a: Sample.SpecInfo, b: Sample.SpecInfo) -> Sample.SpecInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)  # config-access-exempt: attribute selected at runtime from field

    return _create_with_all_fields(
        Sample.SpecInfo,
        spec_num_correct_drafts=_merge_plus_value("spec_num_correct_drafts"),
        spec_num_proposed_drafts=_merge_plus_value("spec_num_proposed_drafts"),
        spec_verify_ct=_merge_plus_value("spec_verify_ct"),
        completion_tokens=_merge_plus_value("completion_tokens"),
    )


def _merge_prefix_cache_info(a: Sample.PrefixCacheInfo, b: Sample.PrefixCacheInfo) -> Sample.PrefixCacheInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)  # config-access-exempt: attribute selected at runtime from field

    return _create_with_all_fields(
        Sample.PrefixCacheInfo,
        cached_tokens=_merge_plus_value("cached_tokens"),
        total_prompt_tokens=_merge_plus_value("total_prompt_tokens"),
    )


def _create_with_all_fields(cls, **kwargs):
    expected = {f.name for f in fields(cls)}
    actual = set(kwargs.keys())
    assert (
        expected == actual
    ), f"{cls.__name__} field mismatch. Missing: {expected - actual}, Extra: {actual - expected}"
    return cls(**kwargs)


def _startswith(*, short, long) -> bool:
    if isinstance(short, str) and isinstance(long, str):
        return long.startswith(short)
    if isinstance(short, list) and isinstance(long, list):
        return (len(long) >= len(short)) and (long[: len(short)] == short)
    raise NotImplementedError


def _len_or_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, str)):
        return {"type": type(value).__name__, "len": len(value)}
    return value


def sample_text_preview(sample: Sample, max_chars: int = 512) -> str:
    text = (str(sample.prompt) + sample.response).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated chars={len(text) - max_chars}>"


def reward_log_summary(reward: Any) -> Any:
    """Summarize a reward (which for OPD scoring can be a large nested dict of logprobs)
    into shapes/lengths instead of dumping the whole thing into the log."""
    if not isinstance(reward, dict):
        return _len_or_value(reward)

    summary: dict[str, Any] = {}
    for key, value in reward.items():
        if not isinstance(value, dict):
            summary[key] = _len_or_value(value)
            continue

        entry: dict[str, Any] = {"keys": list(value.keys())}
        meta_info = value.get("meta_info")
        if isinstance(meta_info, dict):
            entry["meta_info"] = {
                meta_key: _len_or_value(meta_info[meta_key])
                for meta_key in (
                    "id",
                    "finish_reason",
                    "prompt_tokens",
                    "weight_version",
                    "input_token_logprobs",
                    "input_token_ids_logprobs",
                    "input_top_logprobs",
                )
                if meta_key in meta_info
            }
        summary[key] = entry
    return summary
