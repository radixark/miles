"""The reader of Megatron-Bridge PEFT ``iter_N/`` checkpoints (torch_dist), loaded straight into a single-LoRA run."""

import re
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.lora.utils import (
    OPTIMIZER_KEY,
    _distributed_optimizers,
    _zero_init_empty_hdo_states,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import AdapterCheckpointMembers
from miles.backends.training_utils.checkpoint.adapter.adapter_state import agree_on_failure_context, gloo_group
from miles.backends.training_utils.checkpoint.adapter.base_model_check import assert_same_base_model
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.parallel import get_parallel_state

_BRIDGE_LORA_TYPES = {
    "megatron.bridge.peft.lora.LoRA": "lora",
    "megatron.bridge.peft.canonical_lora.CanonicalLoRA": "canonical_lora",
}


def _yaml_dtype(value) -> str | None:
    # run_config.yaml writes a dtype as {_target_: torch.bfloat16, _call_: false}
    return None if value is None else value["_target_"]


def assert_bridge_run_config_matches(
    config: dict, run: AdapterParametrization, *, params_dtype: torch.dtype, load_optimizer: bool
) -> None:
    """A Megatron-Bridge run_config.yaml must describe the LoRA this run trains, and resumable Adam state."""
    peft = config["peft"]
    lora_type = _BRIDGE_LORA_TYPES.get(peft["_target_"])
    assert lora_type is not None, f"Megatron-Bridge PEFT {peft['_target_']} is neither LoRA nor CanonicalLoRA"
    # .get: a field newer than the writing Bridge defaults to Bridge's behaviour before it
    source = {
        "lora_type": lora_type,
        "lora_rank": peft["dim"],
        "experts_shared_outer_loras": peft.get("experts_shared_outer_loras", False),
    }
    run_fields = asdict(run)
    for field, value in source.items():
        assert value == run_fields[field], f"the source has {field}={value!r}; this run has {run_fields[field]!r}"
    assert peft["dropout"] == 0, f"LoRA dropout {peft['dropout']} is not supported"
    assert not peft["exclude_modules"], f"exclude_modules {peft['exclude_modules']} is not supported"
    assert not peft.get("normalize_moe_lora", False), "normalize_moe_lora makes expert ranks depend on the saving ETP"
    # CanonicalLoRA has neither field
    if lora_type == "lora":
        assert not peft["a2a_experimental"], "a2a_experimental LoRA is not supported"
        assert _yaml_dtype(peft["lora_dtype"]) in (
            None,
            str(params_dtype),
        ), f"lora_dtype {_yaml_dtype(peft['lora_dtype'])} differs from the model's {params_dtype}"
    expert_targets = any(".experts." in module for module in run.target_modules)
    if expert_targets and config["model"]["moe_grouped_gemm"] and not run.experts_shared_outer_loras:
        # Bridge shares one adapter across grouped experts by default; sequential experts always get one each,
        # and a grouping that differs from this run's fails the adapter key check
        assert not peft.get("share_expert_adapters", True), "share_expert_adapters=True is not supported"
    if load_optimizer:
        optimizer = config["optimizer"]
        assert (
            optimizer["optimizer"] == "adam"
        ), f"optimizer {optimizer['optimizer']} has no Adam state; pass --no-load-optim"
        assert optimizer[
            "use_distributed_optimizer"
        ], "the source did not use the DistributedOptimizer; pass --no-load-optim"
        assert not optimizer["use_precision_aware_optimizer"], "precision-aware optimizer state is not supported"
        for field in ("main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
            assert _yaml_dtype(optimizer[field]) in (None, "torch.float32"), f"{field} must be fp32"
        assert config["checkpoint"]["save_optim"], "the source saved no optimizer state; pass --no-load-optim"


_BRIDGE_BASE_REMEDY = (
    "download it into the local HF cache (`hf download <id>`), or set checkpoint.hf_source_path in the source's "
    "run_config.yaml to a local HF dir of it"
)


def bridge_base_model_candidates(config: dict) -> tuple[str | None, str | None]:
    """The source's base candidates, best first: pretrained_checkpoint, then the HF template Bridge's
    _resolve_hf_source picks."""
    checkpoint = config["checkpoint"]
    # Bridge's precedence: the first non-empty one, even when it is no HF dir here
    template = (
        checkpoint.get("hf_source_path")
        or config["model"].get("hf_model_id")
        or config["tokenizer"]["tokenizer_model"]
    )
    return checkpoint["pretrained_checkpoint"], template


def torch_dist_base_dir(pretrained_checkpoint: str | None) -> Path | None:
    """The torch_dist checkpoint at ``pretrained_checkpoint``, or at the iteration its Megatron tracker names; None
    when it is no torch_dist on this machine."""
    if not pretrained_checkpoint or not Path(pretrained_checkpoint).is_dir():
        return None
    root = Path(pretrained_checkpoint)
    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        iteration = tracker.read_text().strip()
        root = root / ("release" if iteration == "release" else f"iter_{int(iteration):07d}")
    return root if dist_checkpointing.check_is_distributed_checkpoint(str(root)) else None


def _is_base_check_tensor(key: str, tensor) -> bool:
    # a full fine-tune changes the norms, a merged LoRA the projections; a chunk's layer 0 is its first local layer
    return (
        isinstance(tensor, ShardedTensor)
        and ".adapter." not in key
        and not is_float8tensor(tensor.data)
        and ("norm" in key or re.search(r"(^|\.)layers\.0\.self_attention\.", key) is not None)
    )


def assert_same_base_as_torch_dist(source: str, chunks, base_dir: Path) -> None:
    """Collective: every norm weight and layer-0 attention weight this rank's model holds unquantized equals the
    torch_dist base checkpoint's, bitwise. The in-memory model is the run's base, so a base fine-tuned in Megatron
    after its HF conversion is caught. Not detected: a LoRA merged only into MLP or expert weights, or only into
    layers past each chunk's first."""
    held, requested = {}, {}
    with agree_on_failure_context():
        metadata = dict(dist_checkpointing.load_content_metadata(str(base_dir)) or {})
        metadata["dp_cp_group"] = get_parallel_state().intra_dp_cp.group
        for index, chunk in enumerate(chunks):
            compared = {
                k: t for k, t in chunk.sharded_state_dict(metadata=metadata).items() if _is_base_check_tensor(k, t)
            }
            held |= {(index, k): t.data for k, t in compared.items()}
            # loading writes into the requested tensors: empty copies, never the model's own parameters
            requested[f"model{index}"] = {k: replace(t, data=torch.empty_like(t.data)) for k, t in compared.items()}
        assert any("norm" in k for _, k in held), "this run's model holds no norm weights to compare its base by"
        saved = dcp.FileSystemReader(base_dir).read_metadata().state_dict_metadata
        missing = sorted({t.key for sd in requested.values() for t in sd.values()} - saved.keys())
        assert not missing, (
            f"{source}'s base checkpoint {base_dir} cannot be compared with this run's model: it lacks e.g. "
            f"{missing[:3]}"
        )
    with agree_on_failure_context():
        loaded = dist_checkpointing.load(requested, str(base_dir))
        differing = [k for (index, k), data in held.items() if not torch.equal(loaded[f"model{index}"][k], data)]
        assert not differing, (
            f"{source} was trained on another base model than this run's --hf-checkpoint: its base checkpoint "
            f"{base_dir} differs in e.g. {differing[:3]}"
        )


class MegatronBridgeCheckpointLoader:
    """A Megatron-Bridge PEFT ``iter_N/`` (torch_dist), read straight into this run's model and DistributedOptimizer
    under any TP / PP / EP / DP layout. The source's iteration, scheduler and RNG are not resumed."""

    def __init__(
        self, model, optimizer, members: AdapterCheckpointMembers, run: AdapterParametrization, load_optimizer: bool
    ):
        self._model, self._optimizer, self._members = model, optimizer, members
        self._run, self._load_optimizer = run, load_optimizer

    def parametrization(self, path: Path) -> tuple[AdapterParametrization, bool]:
        """Collective: every check against the source, reading it only; ``apply`` then loads what it requested."""
        chunks = unwrap_model(self._model)
        names = ["model"] if len(chunks) == 1 else [f"model{i}" for i in range(len(chunks))]
        config, metadata, model_state_dicts, adapter, base_dir = self._read_and_check_source(path, chunks)
        self._assert_same_torch_dist_base(path, chunks, base_dir)
        has_optimizer_state = self._check_adapter_keys(path, adapter)
        self._checked = (path, names, chunks, model_state_dicts, adapter, metadata)
        return replace(self._run, lora_alpha=float(config["peft"]["alpha"])), has_optimizer_state

    def prepare(self, path: Path) -> tuple:
        # the source was read and checked with parametrization; nothing rank-local is left before the load
        assert self._checked[0] == path, f"prepare({path}) after checking {self._checked[0]}"
        return self._checked

    def apply(self, prepared: tuple) -> None:
        path, names, chunks, model_state_dicts, adapter, metadata = prepared
        requested = dict(zip(names, adapter, strict=True))
        if self._load_optimizer:
            requested["optimizer"] = self._optimizer_state_to_load(names, model_state_dicts, metadata)
        self._load(path, names, chunks, requested)
        # collective; asserts each bf16 adapter parameter is its fp32 master
        self._members.export_adapter_state(0)

    def _read_and_check_source(self, path: Path, chunks):
        """Collective: the source's run_config.yaml checked against this run, its content metadata, and this run's
        sharded state dicts with their adapter subsets."""
        with agree_on_failure_context():
            config = yaml.safe_load((path / "run_config.yaml").read_text())
            assert config["checkpoint"]["ckpt_format"] == "torch_dist", f"{path} is not a torch_dist checkpoint"
            assert_bridge_run_config_matches(
                config, self._run, params_dtype=chunks[0].config.params_dtype, load_optimizer=self._load_optimizer
            )
            base_dir = torch_dist_base_dir(config["checkpoint"]["pretrained_checkpoint"])
            if base_dir is None:
                assert_same_base_model(
                    str(path), bridge_base_model_candidates(config), self._run.base_model, remedy=_BRIDGE_BASE_REMEDY
                )
            metadata = dict(dist_checkpointing.load_content_metadata(str(path)) or {})
            if self._load_optimizer:
                _assert_reshardable_optimizer_state(path, metadata)
            metadata["dp_cp_group"] = get_parallel_state().intra_dp_cp.group
            model_state_dicts = [chunk.sharded_state_dict(metadata=metadata) for chunk in chunks]
            adapter = [
                {k: v for k, v in sd.items() if ".adapter." in k and "_extra_state" not in k}
                for sd in model_state_dicts
            ]
        return config, metadata, model_state_dicts, adapter, base_dir

    def _assert_same_torch_dist_base(self, path: Path, chunks, base_dir: Path | None) -> None:
        """Collective: compares the model with a torch_dist pretrained_checkpoint every rank sees."""
        # a base visible on some nodes only would leave the others outside the collective load
        base_dirs = [None] * dist.get_world_size()
        dist.all_gather_object(base_dirs, base_dir, group=gloo_group())
        assert len(set(base_dirs)) == 1, f"ranks resolve {path}'s pretrained_checkpoint differently: {set(base_dirs)}"
        if base_dir is not None:
            assert_same_base_as_torch_dist(str(path), chunks, base_dir)

    def _check_adapter_keys(self, path: Path, adapter) -> bool:
        """Collective: the source holds exactly this run's adapters; returns whether it holds optimizer state."""
        # pipeline stages hold different layers; the checkpoint holds them all
        expected = [None] * dist.get_world_size()
        dist.all_gather_object(expected, {t.key for sd in adapter for t in nested_values(sd)}, group=gloo_group())
        expected = set().union(*expected)
        with agree_on_failure_context():
            keys = list(dcp.FileSystemReader(path).read_metadata().state_dict_metadata)
            has_optimizer_state = any(OPTIMIZER_KEY.match(k) for k in keys)
            saved = {k for k in keys if ".adapter." in k and "_extra_state" not in k and not OPTIMIZER_KEY.match(k)}
            assert (
                expected == saved
            ), f"{path} holds other adapters than this run's model, e.g. {sorted(expected ^ saved)[:3]}"
        return has_optimizer_state

    def _optimizer_state_to_load(self, names, model_state_dicts, metadata) -> dict:
        """Collective: the DistributedOptimizer's sharded state to load into, built once its offloaded state exists."""
        with agree_on_failure_context():
            # sharded_state_dict(is_loading=True) runs Megatron's load_state_dict, which would dummy-step an empty
            # HDO state
            _zero_init_empty_hdo_states(_distributed_optimizers(self._optimizer))
            state = self._optimizer.sharded_state_dict(
                dict(zip(names, model_state_dicts, strict=True)), is_loading=True, metadata=metadata
            )
        return state

    def _load(self, path: Path, names, chunks, requested: dict) -> None:
        """Collective: loads the adapters, and the Adam state or masters re-cast from them."""
        with agree_on_failure_context():
            # a rank-local failure inside load's own collectives still hangs; one after them is agreed below
            loaded = dist_checkpointing.load(requested, str(path))
            for name, chunk in zip(names, chunks, strict=True):
                result = chunk.load_state_dict(loaded[name], strict=False)
                assert not result.unexpected_keys, f"unexpected adapter keys {result.unexpected_keys[:3]}"
            if self._load_optimizer:
                self._load_optimizer_state(path, loaded["optimizer"])
            else:
                self._optimizer.reload_model_params()

    def _load_optimizer_state(self, path: Path, state: dict) -> None:
        # the run owns the schedule: only state and step come from the source, not its lr or weight decay
        run_hyperparameters = [
            {k: v for k, v in g.items() if k not in ("params", "step")} for g in self._optimizer.param_groups
        ]
        try:
            self._optimizer.load_state_dict(state)
        except KeyError as exc:
            # the DistributedOptimizer matches param groups by (wd_mult, lr_mult, is_expert_parallel,
            # is_decoupled_lr) and raises a bare KeyError of the missing key
            raise ValueError(f"{path} has no optimizer param group {exc} of this run's; pass --no-load-optim") from exc
        for group, hyperparameters in zip(self._optimizer.param_groups, run_hyperparameters, strict=True):
            group.update(hyperparameters)


def _assert_reshardable_optimizer_state(path: Path, metadata: dict) -> None:
    sharding = metadata.get("distrib_optim_sharding_type")
    assert sharding is not None, (
        f"{path} records no optimizer sharding type (written by an older Megatron), so its Adam state "
        "cannot be resharded; pass --no-load-optim"
    )
    assert sharding in DistributedOptimizer.checkpoint_fully_reshardable_formats, (
        f"{path}'s optimizer state is {sharding}, which only loads under its saving layout; re-save it "
        "from Megatron-Bridge with dist_ckpt_optim_fully_reshardable=True, or pass --no-load-optim"
    )
