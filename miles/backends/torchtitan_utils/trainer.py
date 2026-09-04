"""torchtitan's Trainer, adopted whole as the training black box.

miles does not assemble torchtitan internals: ``Trainer.__init__`` already owns
the entire build (spec -> config tree -> ParallelDims -> parallelize/pipelining
-> init -> optimizers -> LR -> checkpointer -> loss wiring, including handing
the loss to the PP schedule), and ``forward_backward_step`` already hides the
PP/non-PP split behind one call (the seam torchtitan committed to for
integrators in pytorch/torchtitan#3856). What miles adds here is only the RL
coupling:

* ``build_trainer_config`` -- one ``Trainer.Config`` tree from miles args. The
  config tree is the program: the HF checkpoint load is
  ``checkpoint.initial_load_in_hf`` (the checkpointer resolves weights from
  ``hf_assets_path``), the RL loss is ``config.loss`` (so the trainer wires it
  into the pipeline schedule itself), and the dataloader is an empty stub
  because the RL loop feeds microbatches directly.
* ``RLLossAdapter`` -- a ``BaseLoss`` whose targets are microbatch indices: the
  schedule only transports tensors, so each target names the miles batch the
  real loss closure runs on. One class serves train (loss + log dict) and
  forward-only (log-prob compute) via an armed mode.
* ``TitanTrainer`` -- the Trainer subclass. It adds nothing to construction;
  it exposes ``step_runner()`` (the shared loop's per-optimizer-step protocol)
  and forward-only passes, both delegating to the trainer's own
  ``forward_backward_step`` / ``pp_schedule``.
* ``hf_weights`` -- HF-named full tensors for the rollout engines, via the
  family's state-dict adapter (dp/tp gathered, pp broadcast).

Like ``megatron_utils`` with megatron.core, this module imports torchtitan at
module scope: it is only ever imported by the torchtitan backend, where
torchtitan is a hard dependency. The version bridges between released torch
and the nightly APIs torchtitan tracks live in ``compat`` and install at
import time, before any torchtitan object is built.
"""

import importlib
import inspect
import json
import logging
import os
from argparse import Namespace
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist

from miles.backends.torchtitan_utils import compat

compat.install()

from torch.distributed._functional_collectives import all_gather_single_autograd  # noqa: E402

from torchtitan.components import checkpoint as titan_checkpoint  # noqa: E402
from torchtitan.components.dataloader import BaseDataLoader  # noqa: E402
from torchtitan.components.loss import BaseLoss  # noqa: E402
from torchtitan.components.optimizer import ParamGroupConfig  # noqa: E402
from torchtitan.distributed import utils as titan_dist_utils  # noqa: E402
from torchtitan.distributed.activation_checkpoint import FullAC  # noqa: E402
from torchtitan.distributed.context_parallel import cp_shard  # noqa: E402
from torchtitan.trainer import Trainer  # noqa: E402

from miles.backends.fsdp_utils.dtensor import gather_full_param  # noqa: E402
from miles.backends.torchtitan_utils import routing_replay  # noqa: E402
from miles.backends.torchtitan_utils.parallel import parallel_dims_from_config  # noqa: E402
from miles.backends.training_utils.torch_native_loop import StepMetrics  # noqa: E402

logger = logging.getLogger(__name__)

# FlexAttention's block size: torchtitan's context-parallel sharding requires the
# query length to be divisible by cp_degree * this.
_FLEX_BLOCK = 128


def _gather_over_mesh(tensor: torch.Tensor, mesh) -> torch.Tensor:
    """All-gather along dim 0 over ``mesh``, carrying gradients.

    The logit gather needs the gradient path: the loss is taken on the full
    sequence, so each rank's shard has to get its slice of the gradient back,
    and a plain all_gather yields a tensor with no autograd history that
    backward then refuses. torch's functional-collectives variant gathers along
    dim 0 only, which is why callers transpose.
    """
    return all_gather_single_autograd(tensor, 0, mesh.get_group())


def _probe(label: str) -> None:
    """One line of device memory, gated on an env var.

    The weight stream is where a rank's residency peaks, and the difference
    between a spike at conversion time and a climb across the stream is what
    separates "the adapter materializes everything up front" from "the
    consumer accumulates".
    """
    if os.environ.get("MILES_TITAN_MEM_PROBE") != "1":
        return
    logger.info(
        f"[mem-probe] {label}: allocated={torch.cuda.memory_allocated() / 2**30:.2f}GB "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GB "
        f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB"
    )


def resolve_model_spec(args: Namespace):
    """The single model entry point: ``torchtitan.models.<name>.model_registry``."""
    module_name = f"torchtitan.models.{args.titan_model_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"--titan-model-name {args.titan_model_name!r} does not resolve to a torchtitan "
            f"model package ({module_name}). Check the pinned torchtitan checkout."
        ) from e
    registry = getattr(module, "model_registry", None)
    if registry is None:
        raise ValueError(f"{module_name} exposes no model_registry(); cannot build a ModelSpec")
    return registry(args.titan_model_flavor, attn_backend=args.titan_attn_backend)


def build_trainer_config(args: Namespace, *, hf_assets_path: str, lr_total_steps: int, dump_subdir: str):
    """One Trainer.Config tree from miles arguments.

    The parallelism section is filled from miles args with the FSDP degree
    left at -1: torchtitan's own ``ParallelDims.from_config`` infers it, and
    the same dims math sizes the batch fields below.
    """
    if args.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.optimizer!r}")

    config = Trainer.Config()
    config.model_spec = resolve_model_spec(args)
    if args.titan_num_layers:
        # Loading a few-layer cutdown of a large checkpoint: the flavor carries
        # the full model's dimensions and only its depth is trimmed, which must
        # match the checkpoint exactly. Per-block init scaling stays computed
        # for the full depth -- harmless when real weights land on top, but it
        # makes a from-scratch run with this flag meaningless.
        available = len(config.model_spec.model.layers)
        if args.titan_num_layers > available:
            raise ValueError(
                f"--titan-num-layers {args.titan_num_layers} exceeds the "
                f"{args.titan_model_flavor} flavor's {available} blocks"
            )
        config.model_spec.model.layers = config.model_spec.model.layers[: args.titan_num_layers]
        logger.info(
            f"Truncated {args.titan_model_flavor} to {args.titan_num_layers} of {available} blocks"
        )
    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.path.join(args.save or "./outputs", "torchtitan", dump_subdir)

    # Parallelism settings pass through verbatim: the miles flags are
    # torchtitan's own ParallelismConfig fields (names, defaults, semantics),
    # so the config tree carries exactly what a torchtitan user would write.
    config.parallelism.data_parallel_replicate_degree = args.titan_data_parallel_replicate_degree
    config.parallelism.data_parallel_shard_degree = args.titan_data_parallel_shard_degree
    config.parallelism.tensor_parallel_degree = args.titan_tensor_parallel_degree
    config.parallelism.pipeline_parallel_degree = args.titan_pipeline_parallel_degree
    config.parallelism.context_parallel_degree = args.titan_context_parallel_degree
    config.parallelism.expert_parallel_degree = args.titan_expert_parallel_degree
    if args.titan_pipeline_parallel_schedule:
        config.parallelism.pipeline_parallel_schedule = args.titan_pipeline_parallel_schedule
    config.parallelism.pipeline_parallel_microbatch_size = 1
    parallel_dims = parallel_dims_from_config(config.parallelism)
    dp_size = parallel_dims.dp_replicate * parallel_dims.dp_shard

    config.training.seq_len = args.titan_seq_len
    # One miles microbatch (a packed (1, seq) document batch) is one trainer
    # "sample": local_batch_size is the per-rank microbatch count of one
    # optimizer step, so the PP schedule (built from it) matches the RL loop.
    config.training.local_batch_size = max(args.global_batch_size // dp_size // args.micro_batch_size, 1)
    config.training.global_batch_size = config.training.local_batch_size * dp_size
    config.training.steps = max(lr_total_steps, 1)
    config.training.max_norm = args.clip_grad
    config.training.disable_cuda_graphs = True  # microbatch shapes vary across rollouts
    if args.fp16:
        config.training.dtype = "float16"

    # One catch-all group: OptimizersContainer asserts every trainable param is
    # claimed by exactly one group.
    config.optimizer.param_groups = [
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs={
                "lr": args.lr,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            },
        )
    ]

    config.loss = RLLossAdapter.Config()
    config.dataloader = EmptyDataLoader.Config()
    config.checkpoint = TiedCheckpointManager.Config()
    # miles' existing flag maps onto titan's own AC component; None means off.
    config.activation_checkpoint = FullAC.Config() if getattr(args, "gradient_checkpointing", False) else None
    config.debug.seed = args.seed

    # The checkpointer must be enabled for the initial load: with no native
    # checkpoint under dump_folder it falls through to the HF assets load
    # (from_hf via the family's state-dict adapter).
    config.checkpoint.enable = True
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.initial_load_in_hf = True

    # miles owns experiment tracking; titan's metrics stay console-only.
    config.metrics.enable_tensorboard = False
    config.metrics.enable_wandb = False
    config.validator.enable = False
    return config


class RLLossAdapter(BaseLoss):
    """Trampoline between the schedule's (pred, target) and miles' RL loss.

    Targets are microbatch-index tensors: the schedule only transports
    tensors, and the RL loss needs the whole miles batch (advantages, old log
    probs, masks), which stays outside torchtitan. ``arm`` sets the batches
    and closure for the next step; in eval mode the closure result is stashed
    and a zero scalar returned (the schedule requires a loss).

    Results are keyed by microbatch index rather than appended: the schedule
    may invoke the loss outside the scheduled microbatches (its first step
    runs a backward-metadata inference call), which upstream's pure losses
    never notice. Keying makes those calls idempotent -- the scheduled pass
    overwrites, and exactly one result per microbatch survives.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config=None):
        self.config = config
        self._batches: list | None = None
        self._closure: Callable | None = None
        self._mode = "train"
        self._results: dict[int, object] = {}
        self._cp_mesh = None
        self._cp_balancer = "headtail"
        self._cp_restore: dict[int, torch.Tensor] = {}

    def set_context_parallel(self, mesh, balancer_type: str) -> None:
        """Gather CP-sharded logits before the RL loss sees them.

        Context parallelism is internal to the trainer: miles' loss hub is
        handed full-length logits, so the memory it needs is the same as at
        cp=1 while attention keeps CP's shorter sequences.
        """
        self._cp_mesh = mesh
        self._cp_balancer = balancer_type
        self._cp_restore = {}

    def _restore_indices(self, seq_len: int, device) -> torch.Tensor:
        """Where each sequence position ends up, asked of torchtitan directly.

        Rather than reproduce the load-balancing permutation, this shards a
        vector of positions through the same ``cp_shard`` the trainer shards its
        inputs with and gathers the result: slot i of the gathered logits then
        holds position ``order[i]``, and the inverse of that is the permutation
        to undo. Nothing here names a balancer, so torchtitan stays the single
        source of truth for the layout.

        Cached per length rather than computed once: without pipeline
        parallelism microbatches keep their own lengths, so one permutation
        cannot cover them all.
        """
        cached = self._cp_restore.get(seq_len)
        if cached is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            (local,), _ = cp_shard(self._cp_mesh, (positions,), None, self._cp_balancer)
            order = _gather_over_mesh(local.flatten(), self._cp_mesh)
            cached = order.argsort()
            self._cp_restore[seq_len] = cached
        return cached

    def arm(self, batches: list, closure: Callable, mode: str) -> None:
        self._batches, self._closure, self._mode = batches, closure, mode
        self._results = {}

    def collect(self) -> list:
        missing = [i for i in range(len(self._batches)) if i not in self._results]
        if missing:
            raise RuntimeError(f"the schedule never ran microbatch(es) {missing}")
        return [self._results[i] for i in range(len(self._batches))]

    def _gather_context_parallel(self, pred: torch.Tensor) -> torch.Tensor:
        """All-gather sequence-sharded logits and undo the CP permutation.

        The gather has to carry gradients: the loss is taken on the full
        sequence, so each rank's shard needs its slice of the gradient back.
        Plain ``dist.all_gather`` produces a tensor with no autograd history
        and ``loss.backward()`` fails on it. The functional-collectives variant
        gathers along dim 0 only, hence the transpose.
        """
        gathered = _gather_over_mesh(pred.transpose(0, 1).contiguous(), self._cp_mesh)
        gathered = gathered.transpose(0, 1)
        restore = self._restore_indices(gathered.shape[1], gathered.device)
        return gathered.index_select(1, restore)

    def __call__(self, pred, target, global_valid_tokens=None, **kwargs):
        from torch.distributed.tensor import DTensor

        if isinstance(pred, DTensor):
            # Under TP titan shards the lm_head output over the vocab dim
            # (Shard(-1)) -- exactly the Megatron vocab-parallel dialect miles'
            # loss hub speaks (its softmax reduces over parallel_state.tp). So
            # the loss gets the local shard; gathering to full vocab instead
            # would double-count the softmax denominator, shifting every
            # log-prob by -ln(tp).
            for placement in pred.placements:
                if not (placement.is_shard() and placement.dim in (pred.ndim - 1, -1)):
                    raise RuntimeError(
                        f"expected vocab-sharded logits (Shard({pred.ndim - 1})), got {pred.placements}"
                    )
            pred = pred.to_local()

        # After the DTensor unwrap, never before: the CP gather is a plain
        # collective over the cp mesh, and under TP the logits arrive as a
        # DTensor whose local shard is what actually has to be gathered.
        if self._cp_mesh is not None:
            pred = self._gather_context_parallel(pred)

        # Any element identifies the batch (see _microbatch_inputs); under CP
        # this rank holds only a slice of the target.
        index = int(target.flatten()[0])
        batch = self._batches[index]
        if self._mode == "train":
            loss, log_dict = self._closure(pred, batch)
            self._results[index] = log_dict
            return loss, {}
        self._results[index] = self._closure(pred, batch)
        return torch.zeros((), device=pred.device, dtype=torch.float32), {}


class EmptyDataLoader(BaseDataLoader):
    """The RL loop feeds microbatches directly; the trainer's own dataloader
    is never iterated and checkpoints no state."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        pass

    def __init__(self, config: Config, **kwargs):
        self.config = config

    def __iter__(self):
        return iter(())

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class TiedCheckpointManager(titan_checkpoint.CheckpointManager):
    """CheckpointManager whose HF load survives tied checkpoints.

    torchtitan flavors qwen3_5 with a separate ``lm_head`` while the HF
    checkpoint ties it to the embedding and ships no ``lm_head.weight``;
    upstream ``dcp_load`` requests every exported key and dies on the missing
    one. The from_hf branch below is upstream's, plus: keys the checkpoint
    does not ship are dropped from the request (the adapter's ``from_hf``
    reconstructs them), and when the dropped key is the tied lm_head on a rank
    that owns no embedding (a PP last stage), the checkpoint's embedding is
    requested into an lm_head-shaped skeleton so the reconstruction has a
    source.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(titan_checkpoint.CheckpointManager.Config):
        pass

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        if not from_hf:
            return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)

        assert self.sd_adapter is not None
        hf_state = self.sd_adapter.to_hf(state_dict)
        index_mapping = getattr(self.sd_adapter, "fqn_to_index_mapping", None)
        if index_mapping:
            available = set(index_mapping)
            dropped = sorted(k for k in hf_state if k not in available)
            if dropped:
                logger.info(
                    f"HF checkpoint lacks {len(dropped)} exported key(s) (e.g. {dropped[:3]}); "
                    "deferring to the adapter's from_hf reconstruction"
                )
                lm_head_skeleton = hf_state.get("lm_head.weight")
                hf_state = {k: v for k, v in hf_state.items() if k in available}
                if "lm_head.weight" in dropped and lm_head_skeleton is not None:
                    embed_key = next((k for k in available if k.endswith("embed_tokens.weight")), None)
                    if embed_key is not None and embed_key not in hf_state:
                        hf_state[embed_key] = torch.empty_like(lm_head_skeleton)

        titan_checkpoint.dcp.load(
            hf_state,
            storage_reader=self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized),
        )
        self.states[titan_checkpoint.MODEL].load_state_dict(self.sd_adapter.from_hf(hf_state))


class TitanTrainer(Trainer):
    """torchtitan's Trainer with the RL step surface bolted on.

    Construction is entirely the base class. The additions translate the
    shared RL loop's step-runner protocol onto the trainer's own machinery:
    ``forward_backward_step`` (which internally dispatches PP schedule vs
    single model), the optimizer/LR containers, and titan's grad clipping.
    """

    # ----------------------------------------------------------------- data

    def _family_forward_kwargs(self) -> dict:
        """Static per-family forward kwargs (resolved once).

        qwen3_5 dereferences ``special_tokens`` unconditionally, text-only
        included; the ids live in the HF config. These ride input_dict:
        ``post_dataloading_process`` forwards every non-"input" key to the
        model, PP stages included.
        """
        if not hasattr(self, "_family_kwargs"):
            self._family_kwargs = {}
            if "special_tokens" in inspect.signature(self.model_parts[0].forward).parameters:
                hf_cfg = json.load(open(os.path.join(self.config.hf_assets_path, "config.json")))
                self._family_kwargs["special_tokens"] = {
                    "image_id": hf_cfg.get("image_token_id", -1),
                    "video_id": hf_cfg.get("video_token_id", -2),
                }
        return self._family_kwargs

    def padded_length(self, n_tokens: int) -> int:
        """The length a microbatch of ``n_tokens`` is padded to before the model.

        Two reasons to pad, and pipeline parallelism's subsumes the other. Its
        stages' send/recv buffers are shape-inferred once and reused, so every
        microbatch of the whole run must have one shape. Context parallelism
        splits the sequence across the cp mesh and flex attention works in
        128-token blocks, so the length has to be a multiple of cp * 128, while
        miles pads only to 128.
        """
        if self.parallel_dims.pp_enabled:
            target = self.config.training.seq_len
            if n_tokens > target:
                raise ValueError(
                    f"packed microbatch of {n_tokens} tokens exceeds --titan-seq-len "
                    f"{target}, which is the fixed shape PP stages exchange"
                )
            return target
        if self.parallel_dims.cp_enabled:
            align = self.parallel_dims.cp * _FLEX_BLOCK
            return n_tokens + (align - n_tokens % align) % align
        return n_tokens

    def align_token_side_channel(self, tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        """Reshape a per-token side channel exactly as the input is reshaped.

        Anything indexed by token position -- the rollout's routing, say -- has
        to follow the tokens: it is padded to the same length and then, under
        context parallelism, sharded across the cp mesh by the same balancer.
        A channel that skipped either step would be read at positions the model
        is not looking at, which is not a crash but a silently wrong answer.

        Takes and returns ``[tokens, ...]`` on whatever device it was given.
        """
        missing = self.padded_length(tensor.shape[0]) - tensor.shape[0]
        if missing:
            pad = torch.full(
                (missing, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device
            )
            tensor = torch.cat([tensor, pad], dim=0)
        if self.parallel_dims.cp_enabled:
            # cp_shard speaks [batch, seq, ...]; the round trip through the
            # device is because the mesh has no CPU backend and these are
            # kilobytes.
            (local,), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (tensor.unsqueeze(0).to(self.device),),
                None,
                self.config.parallelism.context_parallel_load_balancer,
                input_seq_dim=1,
            )
            tensor = local.squeeze(0).to(tensor.device)
        if self.parallel_dims.tp_enabled:
            # torchtitan's tensor-parallel plan shards the block input along the
            # sequence as well, so by the time a layer runs, its share of the
            # tokens is a contiguous chunk of what context parallelism already
            # left this rank. DTensor's Shard splits in rank order.
            mesh = self.parallel_dims.get_mesh("tp")
            tp = mesh.size()
            if tensor.shape[0] % tp:
                raise ValueError(
                    f"a {tensor.shape[0]}-token side channel does not divide across "
                    f"{tp} tensor-parallel ranks"
                )
            tensor = tensor.chunk(tp, dim=0)[dist.get_rank(mesh.get_group())]
        return tensor

    def _microbatch_inputs(self, batches: list) -> tuple[list[dict], list[torch.Tensor]]:
        if self.parallel_dims.pp_enabled:
            expected = self.num_pipeline_parallel_microbatches
            if len(batches) != expected:
                raise ValueError(
                    f"the PP schedule was built for {expected} microbatches per optimizer step "
                    f"but this step has {len(batches)}; global_batch_size / dp / "
                    "micro_batch_size must be constant (no dynamic batch sizing with PP)"
                )

        def _model_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            tokens, positions = batch["tokens"], batch["position_ids"]
            pad = self.padded_length(tokens.shape[1]) - tokens.shape[1]
            if pad:
                # The pad region gets consecutive positions starting at 0, making
                # it a single extra document the loss never reads -- all-zero
                # positions (miles' usual pad fill) would read as thousands of
                # one-token documents, which blows up the per-document state
                # allocation of linear-attention kernels (qwen3_5's
                # GatedDeltaNet).
                tokens = torch.nn.functional.pad(tokens, (0, pad), value=0)
                extra = torch.arange(pad, device=positions.device, dtype=positions.dtype)
                positions = torch.cat([positions, extra.unsqueeze(0)], dim=1)
            return tokens, positions

        input_dicts = []
        for batch in batches:
            tokens, positions = _model_inputs(batch)
            input_dicts.append({"input": tokens, "positions": positions, **self._family_forward_kwargs()})
        # Targets carry the microbatch index, but as a full-length tensor rather
        # than a scalar: context parallelism shards labels along the sequence
        # like everything else, and a 0-d tensor has no dimension to shard. Every
        # element holds the same index, so any shard still identifies the batch.
        labels = [
            torch.full_like(input_dicts[i]["input"], i, dtype=torch.long) for i in range(len(batches))
        ]
        return input_dicts, labels

    # -------------------------------------------------------- RL step surface

    def _pipeline_will_infer_metadata(self, *, has_backward: bool) -> bool:
        """Whether the next schedule call runs its metadata-inference forward.

        The schedule learns the shapes its stages exchange by running one
        forward per stage over microbatch 0, and repeats that whenever the pass
        changes direction -- which in RL is every log-prob-then-train pair. The
        forward is a real one through the model, so anything that consumes state
        per forward (routing replay) has to be told to sit it out. Mirrors
        ``_initialize_pp_stages``' own condition; the attribute check makes a
        rename in torch a loud failure instead of a silent replay corruption.
        """
        schedule = self.pp_schedule
        # Schedules that own one stage per rank track this in the singular,
        # looped ones in the plural.
        for prefix in ("_stage", "_stages"):
            forward_attr = f"{prefix}_forward_initialized"
            backward_attr = f"{prefix}_backward_initialized"
            if hasattr(schedule, forward_attr) and hasattr(schedule, backward_attr):
                break
        else:
            raise RuntimeError(
                f"{type(schedule).__name__} exposes no forward/backward initialization state; "
                "the pipeline schedule's metadata-inference forward can no longer be anticipated"
            )
        if not getattr(schedule, forward_attr):
            return True
        return has_backward != getattr(schedule, backward_attr)

    def run_forward_backward(self, batches, loss_closure: Callable) -> list[dict]:
        """One optimizer step's microbatches through the trainer's own
        forward_backward_step. Under PP only the last stage returns log
        dicts."""
        batches = list(batches)
        self.loss_fn.arm(batches, loss_closure, "train")
        input_dicts, labels = self._microbatch_inputs(batches)
        ones = torch.ones((), device=self.device)
        with routing_replay.consumption_guard(self.model_parts, len(batches)):
            if self.parallel_dims.pp_enabled:
                if self._pipeline_will_infer_metadata(has_backward=True):
                    routing_replay.bypass_schedule_initialization(self.model_parts)
                self.forward_backward_step(input_dict=input_dicts, labels=labels, global_valid_tokens=ones)
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    self.forward_backward_step(input_dict=input_dict, labels=label, global_valid_tokens=ones)
        return self.loss_fn.collect() if self.has_last_stage() else []

    def run_forward(self, batches, compute: Callable) -> list:
        """Forward-only over the microbatches (log probs); mirrors the
        validator's eval path. Under PP only the last stage returns."""
        batches = list(batches)
        self.loss_fn.arm(batches, compute, "eval")
        input_dicts, labels = self._microbatch_inputs(batches)
        with routing_replay.consumption_guard(self.model_parts, len(batches)):
            if self.parallel_dims.pp_enabled:
                arg_mbs, kwarg_mbs, target_mbs = [], [], []
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    arg_mbs.append((inputs,))
                    kwarg_mbs.append(extra)
                    target_mbs.append(label)
                losses = [] if self.pp_has_last_stage else None
                if self._pipeline_will_infer_metadata(has_backward=False):
                    routing_replay.bypass_schedule_initialization(self.model_parts)
                # return_outputs=False matters: the last stage otherwise retains
                # every microbatch's full-vocab logits until the merge -- at RL
                # sequence lengths that alone exceeds device memory. The loss
                # adapter has already consumed each microbatch's logits by then.
                self.pp_schedule.eval(
                    arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                    kwarg_mbs=kwarg_mbs,
                    target_mbs=target_mbs if self.pp_has_last_stage else None,
                    losses=losses,
                    return_outputs=False,
                )
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    pred = self.model_parts[0](inputs, **extra)
                    self.loss_fn(pred, label)
        return self.loss_fn.collect() if self.has_last_stage() else []

    def apply_optimizer_step(self) -> StepMetrics:
        """The optim block of the trainer's train_step, returning what the
        miles loop logs."""
        grad_norm = titan_dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()
        self.step += 1  # the trainer's own step counter, checkpointed as train_state
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        return StepMetrics(grad_norm=float(grad_norm.item()), extra_metrics=self.lr_schedulers.get_metrics())

    def enable_context_parallel_gather(self) -> None:
        """Point the loss adapter at the CP mesh and the sharding it must undo.

        Called once after construction. The adapter recovers the permutation by
        asking torchtitan to shard a vector of positions, so it needs the
        balancer's name only to make that call the same way the trainer does.

        The ptrr balancer is refused: its permutation is derived from the
        attention mask, so it differs per microbatch and cannot be recovered
        from the sequence length alone.
        """
        if not self.parallel_dims.cp_enabled:
            return
        balancer_type = self.config.parallelism.context_parallel_load_balancer
        if balancer_type != "headtail":
            raise ValueError(
                f"context_parallel_load_balancer={balancer_type!r} is not supported yet: only a "
                "data-independent sharding can be undone from the sequence length"
            )
        self.loss_fn.set_context_parallel(self.parallel_dims.get_mesh("cp"), balancer_type)

    def has_last_stage(self) -> bool:
        return (not self.parallel_dims.pp_enabled) or self.pp_has_last_stage

    def step_runner(self) -> "TrainerStepRunner":
        return TrainerStepRunner(self)

    # --------------------------------------------------------------- weights

    def hf_weights(self, *, complete_across_pp: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        """HF-named tensors, materialized one at a time, for the engine push.

        The weight transport requires every rank in an IPC gather group to
        stream the same tensor sequence. dp/tp shards reassemble via
        ``gather_full_param``; under PP each tensor lives on exactly one
        stage, so it is additionally broadcast over the pp mesh -- after which
        every rank yields the identical full stream and the transport stays
        PP-oblivious. One tensor is resident at a time either way.

        An offloaded model comes back to the device for the duration: unlike
        the plain state dicts FSDP streams, titan's fused-QKV save hooks run
        DTensor collectives inside ``state_dict()`` itself, and the meshes
        have no CPU backend. Weights-only occupancy is strictly below the
        training peak, so whenever training fits, this does.
        """
        offloaded = next(self.model_parts[0].parameters()).device.type == "cpu"
        if offloaded:
            for part in self.model_parts:
                part.cuda()
        try:
            yield from self._hf_weights_on_device(complete_across_pp=complete_across_pp)
        finally:
            if offloaded:
                for part in self.model_parts:
                    part.cpu()
                torch.cuda.empty_cache()

    def _hf_weights_on_device(self, *, complete_across_pp: bool) -> Iterator[tuple[str, torch.Tensor]]:
        # The checkpointer only builds its adapter when checkpointing is
        # enabled; weight streaming needs the mapping regardless.
        sd_adapter = getattr(self.checkpointer, "sd_adapter", None)
        if sd_adapter is None:
            sd_adapter = self.config.model_spec.state_dict_adapter(self.model_config, self.config.hf_assets_path)
        _probe("hf_weights: before to_hf")
        local = sd_adapter.to_hf({k: v for part in self.model_parts for k, v in part.state_dict().items()})
        _probe("hf_weights: after to_hf")

        # Which ranks hold which key. Two parallelisms make the export
        # rank-partial: a pipeline stage exports only its own layers, and under
        # expert parallelism the adapter names each rank's experts by their
        # global index, so every rank exports a different slice of them.
        # DTensor.shape is already the global shape, so the metadata describes
        # the post-gather tensor.
        world = dist.get_world_size()
        local_meta = {name: (tuple(t.shape), str(t.dtype)) for name, t in local.items()}
        gathered: list = [None] * world
        dist.all_gather_object(gathered, local_meta)

        if all(meta.keys() == local_meta.keys() for meta in gathered):
            # Every rank exports the same keys: dp/tp/fsdp sharding is internal
            # to each tensor and gather_full_param resolves it.
            for i, name in enumerate(sorted(local)):
                if i % 200 == 0:
                    _probe(f"hf_weights: fast path unit {i}")
                yield name, gather_full_param(local[name])
            return

        owners: dict[str, list[int]] = {}
        specs: dict[str, tuple] = {}
        for rank, meta in enumerate(gathered):
            for name, (shape, dtype) in meta.items():
                owners.setdefault(name, []).append(rank)
                if specs.setdefault(name, (shape, dtype)) != (shape, dtype):
                    raise RuntimeError(f"ranks disagree on the shape/dtype of {name}")

        my_rank = dist.get_rank()
        # Who shares my pipeline stage. Completion is always needed *within* a
        # stage -- expert parallelism has each rank export a different slice of
        # the experts, so no single rank holds a stage's whole set -- and the
        # placement only decides whether it also crosses stages.
        stage_of: list = [None] * world
        pp_mesh = self.parallel_dims.get_optional_mesh("pp")
        my_stage_id = dist.get_rank(group=pp_mesh.get_group()) if pp_mesh is not None else 0
        dist.all_gather_object(stage_of, my_stage_id)
        stage_groups: dict[int, list[int]] = {}
        for rank, stage in enumerate(stage_of):
            stage_groups.setdefault(stage, []).append(rank)
        my_stage = stage_of[my_rank]

        if complete_across_pp:
            audience, broadcast_group = list(range(world)), None
        else:
            # new_group is collective: every rank creates every stage's group.
            groups = {stage: dist.new_group(ranks) for stage, ranks in sorted(stage_groups.items())}
            audience, broadcast_group = stage_groups[my_stage], groups[my_stage]

        audience_set = set(audience)
        names = [name for name in sorted(owners) if audience_set.intersection(owners[name])]
        for i, name in enumerate(names):
            if i % 200 == 0:
                _probe(f"hf_weights: unit {i} of {len(names)}")
            shape, dtype = specs[name]
            holders = [rank for rank in owners[name] if rank in audience_set]
            # Every holder joins the gather -- they are exactly the ranks the
            # tensor's own mesh spans, so the collective is complete. The
            # lowest of them then broadcasts to the ranks that lack it;
            # replicas (data parallelism) hold identical values after a step,
            # so which holder broadcasts does not matter, only that it is
            # agreed on.
            if my_rank in holders:
                tensor = gather_full_param(local[name]).contiguous()
            else:
                tensor = torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]), device=self.device)
            dist.broadcast(tensor, src=holders[0], group=broadcast_group)
            yield name, tensor


class TrainerStepRunner:
    """Adapter from the shared loop's step-runner protocol to the trainer.

    Kept separate because the trainer already has a ``forward_backward_step``
    with torchtitan's own signature; the protocol must not shadow it.
    """

    def __init__(self, trainer: TitanTrainer):
        self.trainer = trainer

    def forward_only_step(self, batches, compute: Callable) -> list:
        return self.trainer.run_forward(batches, compute)

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        return self.trainer.run_forward_backward(batches, loss_closure)

    def zero_grad(self) -> None:
        self.trainer.optimizers.zero_grad(set_to_none=True)

    def apply_step(self) -> StepMetrics:
        return self.trainer.apply_optimizer_step()
