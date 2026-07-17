import logging
from pathlib import Path

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def load_debug_rollout_data(args, rollout_id: int) -> tuple[list[Sample], dict]:
    data, metadata = _load_rollout_data_file(Path(args.load_debug_rollout_data.format(rollout_id=rollout_id)))
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data, metadata


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool, metadata: dict | None = None) -> None:
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO may improve the format
        if evaluation:
            dump_data = dict(
                samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
            )
        else:
            dump_data = dict(
                samples=[sample.to_dict() for sample in data],
            )

        torch.save(dict(rollout_id=rollout_id, metadata=metadata or {}, **dump_data), path)


class RolloutDataInjectionUtil:
    @classmethod
    def should_inject(cls, args, rollout_id: int) -> bool:
        if args.ci_inject_rollout_data_path is None:
            return False
        return rollout_id >= args.ci_inject_rollout_data_start_rollout_id

    @classmethod
    def load(cls, args, rollout_id: int) -> tuple[list[Sample], dict]:
        path = Path(args.ci_inject_rollout_data_path.format(rollout_id=rollout_id))
        assert path.is_file(), f"Recorded rollout data to inject is missing: {path}"
        logger.info(f"CI rollout-data injection: replacing generated data of rollout {rollout_id} with {path}")
        return _load_rollout_data_file(path)

    @classmethod
    def assert_matches_generated(
        cls, args, *, generated: list[Sample], injected: list[Sample], rollout_id: int
    ) -> None:
        min_match_ratio: float = args.ci_inject_rollout_data_min_match_ratio
        assert len(generated) == len(
            injected
        ), f"rollout {rollout_id}: sample count mismatch, generated {len(generated)} vs injected {len(injected)}"

        ratios: list[float] = []
        for index, (generated_sample, injected_sample) in enumerate(zip(generated, injected, strict=True)):
            assert cls._prompt_tokens(generated_sample) == cls._prompt_tokens(injected_sample), (
                f"rollout {rollout_id}: prompt tokens mismatch at sample {index}; "
                "injected recording does not pair with the generated batch"
            )
            ratios.append(cls._response_token_match_ratio(generated_sample, injected_sample))

        mean_ratio = sum(ratios) / len(ratios)
        logger.info(
            f"CI rollout-data injection match for rollout {rollout_id}: "
            f"mean response token match {mean_ratio:.4f} (min {min(ratios):.4f}, "
            f"threshold {min_match_ratio}) over {len(ratios)} samples"
        )
        assert mean_ratio > min_match_ratio, (
            f"rollout {rollout_id}: generated responses match the injected recording at only "
            f"{mean_ratio:.4f} (threshold {min_match_ratio}); the engine "
            "weights likely diverged from the baseline beyond ulp-level drift"
        )

    @classmethod
    def _response_token_match_ratio(cls, a: Sample, b: Sample) -> float:
        response_a = cls._response_tokens(a)
        response_b = cls._response_tokens(b)
        denominator = max(len(response_a), len(response_b))
        if denominator == 0:
            return 1.0
        matched = sum(token_a == token_b for token_a, token_b in zip(response_a, response_b, strict=False))
        return matched / denominator

    @classmethod
    def _prompt_tokens(cls, sample: Sample) -> list[int]:
        return sample.tokens[: len(sample.tokens) - sample.response_length]

    @classmethod
    def _response_tokens(cls, sample: Sample) -> list[int]:
        return sample.tokens[len(sample.tokens) - sample.response_length :]


def _load_rollout_data_file(path: Path) -> tuple[list[Sample], dict]:
    payload = torch.load(path, weights_only=False)
    data = [Sample.from_dict(sample) for sample in payload["samples"]]
    metadata = payload.get("metadata") or {}
    return data, metadata
