"""Where a microbatch's response tokens sit, so that each loss is written once for every platform.

A loss reads its per-token inputs as one row tensor (`rows`), scores the logits (`score`), and reduces with
the layout's `sum_of_sample_mean`; only the layout knows which rows exist. The layout is chosen once per run
(`response_layout_type`):

- `PackedSlices`, on a platform with dynamic shapes: the rows are the responses, each sliced out of the packed
  logits and concatenated, `[sum(R)]` on this CP rank.
- `PaddedRows`, on a platform without dynamic shapes with the torchtitan trainer. TorchTPU compiles every eager
  op per input shape, and RL responses rarely repeat a length, so slicing compiles thousands of programs per
  step. The rows are the `[T]` rows of the padded microbatch, which the trainer rounds to a few lengths:
  prompt and padding rows are masked out, and a per-row divisor reproduces the per-sample mean and its
  gradient. Per-sample tensors stay on the host and reach the device in one copy each; a forward-only pass
  copies its results back without waiting and reads them once, after its last microbatch.

Features that need per-sample tensors (`per_sample`) keep dynamic shapes, so `validate_response_layout_args`
refuses them at startup with `PaddedRows` instead of letting a run take thousands of compiles.
"""

import functools
import logging
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch

from miles import platforms
from miles.backends.training_utils.data.context_parallel import (
    allgather_cp_redistribute,
    get_local_response_loss_masks,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)
from miles.backends.training_utils.data.sampling_mask import get_rollout_sampling_masks
from miles.backends.training_utils.loss.hub.logit_processors import _iter_response_chunks, get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.math_utils import (
    _safe_exp_neg_ppo_kl,
    calculate_log_probs_and_entropy,
    compute_ess_ratio_contribution,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)

# chunks a padded microbatch is scored in, whatever its length
ROW_CHUNKS = 8
_PADDED_ROWS_LOSS_TYPES = ("policy_loss", "score_centering")
_PADDED_ROWS_ADVANTAGE_ESTIMATORS = ("grpo", "reinforce_plus_plus", "reinforce_plus_plus_baseline")
# what PaddedRows assumes about the run: only without CP or PP is every padded row a whole packed sample
_PADDED_ROWS_REQUIREMENTS: dict[str, Callable[[Namespace], bool]] = {
    "--qkv-format thd": lambda args: args.qkv_format == "thd",
    "--context-parallel-size 1": lambda args: args.context_parallel_size == 1,
    "--pipeline-model-parallel-size 1": lambda args: args.pipeline_model_parallel_size == 1,
    f"--advantage-estimator in {_PADDED_ROWS_ADVANTAGE_ESTIMATORS}": lambda args: args.advantage_estimator
    in _PADDED_ROWS_ADVANTAGE_ESTIMATORS,
}
# features whose code reads per-sample tensors, or that PaddedRows does not score
_UNSUPPORTED_FEATURES: dict[str, Callable[[Namespace], bool]] = {
    "--true-on-policy-mode": lambda args: args.true_on_policy_mode,
    "--log-probs-backend fused": lambda args: args.log_probs_backend == "fused",
    "--allgather-cp": lambda args: args.allgather_cp,
    "--use-rollout-sampling-support-replay": lambda args: args.use_rollout_sampling_support_replay,
    "--use-opsm": lambda args: args.use_opsm,
    "--use-tis": lambda args: args.use_tis,
    "--get-mismatch-metrics": lambda args: args.get_mismatch_metrics,
    "--use-opd": lambda args: args.use_opd,
    "--custom-pg-loss-reducer-function-path": lambda args: args.custom_pg_loss_reducer_function_path is not None,
    "--dump-details": lambda args: args.dump_details is not None,
    "multi-LoRA": is_multi_lora_enabled,
}

# fn(chunk logits [n, V], targets [n], per-row side input [n, ...] or None) -> named per-row results
RowFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], dict[str, torch.Tensor]]
# side(sample index, positions of its rows in the response [n]) -> host values for those rows [n, ...]
SideFn = Callable[[int, torch.Tensor], torch.Tensor]


class ResponseLayout(Protocol):
    """The rows a loss works on; `PackedSlices` and `PaddedRows` are the implementations."""

    # where get_rollout_data puts per-sample tensors; None is the accelerator
    sample_device: ClassVar[torch.device | None]
    device: torch.device
    chunk_size: int  # rows a log-prob call scores at a time; -1 is a whole response

    @classmethod
    def build(
        cls,
        args: Namespace,
        batch: RolloutBatch,
        logits: torch.Tensor,
        sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> "ResponseLayout":
        """The layout of one microbatch."""

    @staticmethod
    def finish_forward_pass(results: list[dict]) -> list[dict]:
        """A forward-only pass's results as per-sample values, after its last microbatch."""

    def rows(self, per_sample: Sequence[torch.Tensor], *, fill: float = 0) -> torch.Tensor:
        """Per-sample values on this rank's response rows -> one row tensor on the device."""

    def rows_at_positions(self, side: SideFn, *, fill: float = 0) -> torch.Tensor:
        """Per-sample values built from each row's position in its full response -> one row tensor."""

    def per_sample(self, rows: torch.Tensor) -> list[torch.Tensor]:
        """A row tensor back to per-sample values; only the refused features read these."""

    def loss_masks(self) -> torch.Tensor:
        """This rank's loss masks on the rows; prompt and padding rows are 0."""

    def score(
        self, logits: torch.Tensor, *, with_entropy: bool, entropy_requires_grad: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Log-probs (and entropy) of the response tokens, on the rows."""

    def map_logit_rows(
        self, logits: torch.Tensor, fn: RowFn, side: SideFn | None = None, *, side_fill: float = 0
    ) -> dict[str, torch.Tensor]:
        """`fn(logits, targets, side rows)` over the response rows; results on the rows.

        `PackedSlices` calls it once per response, `PaddedRows` once on all rows (`chunk_size` bounds the
        work `fn` does at a time). `side_fill` fills the side input of rows outside any response.
        """

    def sum_of_sample_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce a row tensor like `get_sum_of_sample_mean` reduces the concatenated responses."""

    def reducer(self, loss_masks: Sequence[torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        """`sum_of_sample_mean` with other per-sample loss masks and the same denominators."""

    def ess_ratio(self, ppo_kl: torch.Tensor) -> torch.Tensor:
        """`compute_ess_ratio_contribution` of a row tensor of per-token log-ratios."""

    def forward_log_probs(self, logits: torch.Tensor, *, with_entropy: bool, use_sampling_mask: bool) -> dict:
        """A forward-only pass's per-sample results, or handles `finish_forward_pass` reads.

        `use_sampling_mask` scores over the rollout's sampling support; the reference pass does not.
        """


def padded_rows_enabled(args: Namespace) -> bool:
    """Whether losses run on `PaddedRows`: the torchtitan trainer pads every microbatch on such a platform."""
    return (
        not platforms.current_platform.dynamic_shapes
        and args.train_backend == "torchtitan"
        and args.loss_type in _PADDED_ROWS_LOSS_TYPES
    )


def validate_response_layout_args(args: Namespace) -> None:
    """On a platform without dynamic shapes, refuse at startup a run `PaddedRows` cannot score.

    The per-sample path it would otherwise need compiles a program per response length on such a platform,
    so the run would be impractically slow rather than wrong. A loss not written on the layout takes that
    path, which is only warned about.
    """
    if platforms.current_platform.dynamic_shapes:
        return
    if not padded_rows_enabled(args):
        logger.warning(
            f"--loss-type {args.loss_type} with --train-backend {args.train_backend} on "
            f"{platforms.current_platform.device_type} takes the per-sample loss path, which compiles a program "
            "per response length"
        )
        return
    problems = []
    if missing := [name for name, met in _PADDED_ROWS_REQUIREMENTS.items() if not met(args)]:
        problems.append(f"needs {', '.join(missing)}")
    if unsupported := [name for name, enabled in _UNSUPPORTED_FEATURES.items() if enabled(args)]:
        problems.append(f"does not implement {', '.join(unsupported)}")
    if problems:
        raise ValueError(
            f"--loss-type {args.loss_type} on {platforms.current_platform.device_type} scores padded "
            f"microbatches, which {' and '.join(problems)}"
        )


def response_layout_type(args: Namespace) -> type[ResponseLayout]:
    return PaddedRows if padded_rows_enabled(args) else PackedSlices


def response_layout(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ResponseLayout:
    """The layout of one microbatch; `sum_of_sample_mean` is the dispatcher's reducer, which `PackedSlices` reuses."""
    return response_layout_type(args).build(args, batch, logits, sum_of_sample_mean)


@dataclass(frozen=True)
class PackedSlices:
    """Rows are this rank's response tokens, concatenated sample by sample: `[sum(R)]` without CP."""

    args: Namespace
    batch: RolloutBatch
    device: torch.device
    dispatcher_sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor] | None

    # get_rollout_data puts per-sample tensors on the accelerator
    sample_device: ClassVar[torch.device | None] = None

    @classmethod
    def build(cls, args, batch, logits, sum_of_sample_mean=None) -> "PackedSlices":
        return cls(args, batch, logits.device, sum_of_sample_mean)

    @functools.cached_property
    def local_loss_masks(self) -> list[torch.Tensor]:
        """Per sample, the loss mask of this rank's rows (its CP slice)."""
        batch = self.batch
        return get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
            self.args.qkv_format,
            batch.get("max_seq_lens"),
        )

    @staticmethod
    def finish_forward_pass(results: list[dict]) -> list[dict]:
        return results

    def rows(self, per_sample, *, fill=0):
        return torch.cat([value.to(self.device) for value in per_sample], dim=0)

    def rows_at_positions(self, side, *, fill=0):
        return self.rows([side(i, positions) for i, positions in enumerate(self._local_positions())], fill=fill)

    def per_sample(self, rows):
        return list(rows.split([mask.size(0) for mask in self.local_loss_masks], dim=0))

    def loss_masks(self):
        return torch.cat(self.local_loss_masks, dim=0).to(device=self.device)

    def score(self, logits, *, with_entropy, entropy_requires_grad):
        result = get_log_probs_and_entropy(
            logits,
            args=self.args,
            unconcat_tokens=self.batch["unconcat_tokens"],
            total_lengths=self.batch["total_lengths"],
            response_lengths=self.batch["response_lengths"],
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=self.batch.get("max_seq_lens"),
            rollout_sampling_mask=self._sampling_masks(),
        )
        entropy = torch.cat(result["entropy"], dim=0) if with_entropy else None
        return torch.cat(result["log_probs"], dim=0), entropy

    def map_logit_rows(self, logits, fn, side=None, *, side_fill=0):
        args, batch = self.args, self.batch
        results: dict[str, list[torch.Tensor]] = {}
        chunks = _iter_response_chunks(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
            include_response_indices=side is not None,
        )
        for i, (chunk, tokens, positions) in enumerate(chunks):
            extra = None if side is None else side(i, torch.as_tensor(list(positions), dtype=torch.long))
            for key, value in fn(chunk, tokens, None if extra is None else extra.to(logits.device)).items():
                results.setdefault(key, []).append(value)
        if args.allgather_cp and get_parallel_state().cp.size > 1:
            allgather_cp_redistribute(
                results,
                logits=logits,
                args=args,
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                max_seq_lens=batch.get("max_seq_lens"),
            )
        return {key: torch.cat(values, dim=0) for key, values in results.items()}

    @functools.cached_property
    def _sum_of_sample_mean(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.dispatcher_sum_of_sample_mean is not None:
            return self.dispatcher_sum_of_sample_mean
        return self.reducer(self.batch["loss_masks"])

    def sum_of_sample_mean(self, x):
        return self._sum_of_sample_mean(x)

    def reducer(self, loss_masks):
        args, batch = self.args, self.batch
        return get_sum_of_sample_mean(
            batch["total_lengths"],
            batch["response_lengths"],
            loss_masks,
            args.backend.calculate_per_token_loss,
            args.qkv_format,
            batch.get("max_seq_lens"),
            denominators=batch.get("rollout_mask_sums"),
        )

    def ess_ratio(self, ppo_kl):
        args, batch = self.args, self.batch
        return compute_ess_ratio_contribution(
            ppo_kl=ppo_kl,
            loss_masks=batch["loss_masks"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            qkv_format=args.qkv_format,
            max_seq_lens=batch.get("max_seq_lens"),
            calculate_per_token_loss=args.backend.calculate_per_token_loss,
        )

    def forward_log_probs(self, logits, *, with_entropy, use_sampling_mask):
        return get_log_probs_and_entropy(
            logits=logits,
            args=self.args,
            unconcat_tokens=self.batch["unconcat_tokens"],
            total_lengths=self.batch["total_lengths"],
            response_lengths=self.batch["response_lengths"],
            with_entropy=with_entropy,
            max_seq_lens=self.batch.get("max_seq_lens"),
            rollout_sampling_mask=get_rollout_sampling_masks(self.batch) if use_sampling_mask else None,
        )

    def _sampling_masks(self):
        return get_rollout_sampling_masks(self.batch) if self.args.use_rollout_sampling_support_replay else None

    def _local_positions(self) -> list[torch.Tensor]:
        """Per sample, the response position of each of this rank's rows."""
        args, batch = self.args, self.batch
        max_seq_lens = batch.get("max_seq_lens")
        return [
            slice_log_prob_with_cp(
                torch.arange(response),
                total,
                response,
                args.qkv_format,
                None if max_seq_lens is None else max_seq_lens[i],
            )
            for i, (total, response) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True))
        ]

    @property
    def chunk_size(self) -> int:
        return self.args.log_probs_chunk_size


@dataclass(frozen=True)
class PaddedRows:
    """Rows are the `[T]` rows of the padded microbatch's `[1, T, V]` logits.

    Row t predicts packed token t + 1, so the response of sample i fills rows
    `[end_i - response_i - 1, end_i - 1)`, where `end_i` is the sample's end in the packed microbatch
    (`get_batch`'s `cu_seqlens_host`).
    """

    args: Namespace
    batch: RolloutBatch
    device: torch.device
    row_index: torch.Tensor  # [sum(R)] int64 on the host: the row of every response token, in sample order
    response_lengths: list[int]
    num_rows: int

    # the padded path keeps per-sample tensors on the host until a microbatch lays them out on its rows
    sample_device: ClassVar[torch.device | None] = torch.device("cpu")

    @classmethod
    def build(cls, args, batch, logits, sum_of_sample_mean=None) -> "PaddedRows":
        assert logits.dim() == 3 and logits.size(0) == 1, f"thd logits are [1, T, V], got {logits.shape}"
        response_lengths = batch["response_lengths"]
        num_rows = logits.size(1)
        # the packer's sample ends; a trailing entry, when present, closes the microbatch's padding
        ends = batch["cu_seqlens_host"][1 : len(response_lengths) + 1]
        if ends and ends[-1] > num_rows:
            raise ValueError(f"{ends[-1]} packed tokens do not fit {num_rows} logit rows")
        row_index = torch.cat(
            [torch.arange(end - response - 1, end - 1) for end, response in zip(ends, response_lengths, strict=True)]
        )
        if row_index.numel() and row_index[0] < 0:
            raise ValueError("the first sample of a microbatch needs at least one prompt token")
        return cls(args, batch, logits.device, row_index, list(response_lengths), num_rows)

    @staticmethod
    def finish_forward_pass(results: list[dict]) -> list[dict]:
        """The pass's per-sample host results, after one synchronize for all of its copies."""
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        return [
            {key: value.per_sample() if isinstance(value, HostRows) else value for key, value in result.items()}
            for result in results
        ]

    @property
    def chunk_size(self) -> int:
        # A fixed number of chunks, not a fixed size: each chunk all-reduces over TP, and on torch_tpu every rank
        # must issue the same number of collectives, while data-parallel ranks pad to different buckets.
        return -(-self.num_rows // ROW_CHUNKS)

    def rows(self, per_sample, *, fill=0):
        """Host per-sample `[R_i, ...]` values -> `[T, ...]` rows on the device (`fill` elsewhere), in one copy."""
        packed = torch.cat([torch.as_tensor(value).to("cpu") for value in per_sample])
        values = torch.full((self.num_rows, *packed.shape[1:]), fill, dtype=packed.dtype)
        values[self.row_index] = packed
        return values.to(self.device)

    def rows_at_positions(self, side, *, fill=0):
        return self.rows([side(i, torch.arange(length)) for i, length in enumerate(self.response_lengths)], fill=fill)

    def per_sample(self, rows):
        return list(rows[self.row_index.to(rows.device)].split(self.response_lengths))

    def loss_masks(self):
        return self._loss_masks

    def score(self, logits, *, with_entropy, entropy_requires_grad):
        log_probs, entropy = calculate_log_probs_and_entropy(
            logits.squeeze(0),
            self._targets,
            get_parallel_state().tp.group,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            chunk_size=self.chunk_size,
            vocab_size=None,
            temperature=self.args.rollout_temperature,
            debug_unified_grad_fused_logprob=self.args.debug_unified_grad_fused_logprob,
        )
        return log_probs.squeeze(-1), entropy

    def map_logit_rows(self, logits, fn, side=None, *, side_fill=0):
        extra = None if side is None else self.rows_at_positions(side, fill=side_fill)
        return fn(logits.squeeze(0), self._targets, extra)

    def sum_of_sample_mean(self, x):
        return (x * self._loss_masks / self._denominators).sum()

    def reducer(self, loss_masks):
        masks = self.rows(loss_masks)
        denominators = self._mean_denominators(loss_masks)
        return lambda x: (x * masks / denominators).sum()

    def ess_ratio(self, ppo_kl):
        """`compute_ess_ratio_contribution` on rows: per-sample sums land in a `[T]` buffer indexed by sample."""
        sample_ids = self.rows([torch.full((length,), i) for i, length in enumerate(self.response_lengths)])
        num_valid = torch.ones(self.num_rows)
        num_valid[: len(self.response_lengths)] = torch.stack(
            [torch.clamp_min(mask.to("cpu", torch.float32).sum(), 1) for mask in self.batch["loss_masks"]]
        )
        num_valid = num_valid.to(self.device)

        is_weights = _safe_exp_neg_ppo_kl(ppo_kl.detach()) * self._loss_masks
        sum_w = torch.zeros_like(is_weights).index_add_(0, sample_ids, is_weights)
        sum_w2 = torch.zeros_like(is_weights).index_add_(0, sample_ids, is_weights * is_weights)
        ess_ratio = (sum_w * sum_w) / (num_valid * torch.clamp_min(sum_w2, 1e-8))
        return (ess_ratio * num_valid).sum() if self.args.backend.calculate_per_token_loss else ess_ratio.sum()

    def forward_log_probs(self, logits, *, with_entropy, use_sampling_mask):
        assert not use_sampling_mask, "validate_response_layout_args refuses sampling-support replay"
        log_probs, entropy = self.score(logits, with_entropy=with_entropy, entropy_requires_grad=False)
        result = {"log_probs": self._to_host(log_probs)}
        if with_entropy:
            result["entropy"] = self._to_host(entropy)
        return result

    def _to_host(self, rows: torch.Tensor) -> "HostRows":
        return HostRows(rows.detach().to("cpu", non_blocking=True), self.row_index, self.response_lengths)

    @functools.cached_property
    def _targets(self) -> torch.Tensor:
        """The token each row predicts: the response tokens on response rows, 0 elsewhere."""
        responses = [
            tokens[tokens.numel() - length :]
            for tokens, length in zip(self.batch["unconcat_tokens"], self.response_lengths, strict=True)
        ]
        return self.rows(responses)

    @functools.cached_property
    def _loss_masks(self) -> torch.Tensor:
        return self.rows(self.batch["loss_masks"])

    @functools.cached_property
    def _denominators(self) -> torch.Tensor:
        return self._mean_denominators(self.batch["loss_masks"])

    def _mean_denominators(self, loss_masks: Sequence[torch.Tensor]) -> torch.Tensor:
        """`[T]` divisors that make `(x * loss_masks / d).sum()` reduce like `get_sum_of_sample_mean`.

        A response row gets its sample's `max(loss_mask.sum(), 1)` (or of `rollout_mask_sums`), every other
        row 1; per-token sums divide by 1. Dividing each row, rather than each sample's sum, keeps the gradient
        of every row exactly what the per-sample reduction gives it.
        """
        if self.args.backend.calculate_per_token_loss:
            return torch.ones(self.num_rows, device=self.device)
        denominators = self.batch.get("rollout_mask_sums")
        if denominators is None:
            denominators = [mask.sum() for mask in loss_masks]
        per_sample = [
            torch.clamp_min(denominator.to("cpu", torch.float32), 1).expand(length)
            for denominator, length in zip(denominators, self.response_lengths, strict=True)
        ]
        return self.rows(per_sample, fill=1)


@dataclass(frozen=True)
class HostRows:
    """`[T]` results on their way to the host; read them only after `finish_forward_pass` synchronized."""

    values: torch.Tensor
    row_index: torch.Tensor
    response_lengths: list[int]

    def per_sample(self) -> list[torch.Tensor]:
        return list(self.values[self.row_index].split(self.response_lengths))
