"""Bridge LoRA's adapter checkpoint members: HF-keyed views of the adapter parameters.

The HF layout of every Megatron adapter element (TP / EP shards, the qkv group permutation, fc1 gate / up,
GDN in_proj, per-expert names) comes from Bridge's own export, run once on index-valued parameters.
"""

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.lora.utils import adapter_named_parameters
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import AdapterCheckpointMembers
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    LocalShardSpec,
    gloo_group,
    raise_if_any_rank_failed,
)
from miles.backends.training_utils.checkpoint.adapter.base_model import assert_same_base_model
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    HfKey,
    fused_components,
    map_peft_keys_to_hf_keys,
    split_fused_lora_b,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.probed_slots import (
    HfPiece,
    ProbedAdapterSlots,
    RowMapping,
    rectangles,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora.hf_lora_targets import HfLoraTargets
from miles.utils.lora.utils import AdapterSpec

# a ChainedOptimizer prefixes each child's keys with chained_<i>. in the formats that are not fully reshardable
OPTIMIZER_KEY = re.compile(r"(chained_\d+\.)?optimizer\.")
_BRIDGE_LORA_TYPES = {
    "megatron.bridge.peft.lora.LoRA": "lora",
    "megatron.bridge.peft.canonical_lora.CanonicalLoRA": "canonical_lora",
}

# a probe value is rank * _RANK_STRIDE + the element's offset in the rank's concatenated adapter parameters;
# float64 holds it exactly
_RANK_STRIDE = 2**40


def _pieces_of(key: str, codes: torch.Tensor, rank: int) -> list[HfPiece]:
    mine = codes // _RANK_STRIDE == rank
    if not mine.any():
        return []
    return [
        HfPiece(key, tuple(codes.shape), tuple(s.start for s in box), (codes[box] % _RANK_STRIDE).cpu())
        for box in rectangles(key, mine)
    ]


def _stack_experts(key: str, pieces_by_expert: dict[int, list[HfPiece]], num_experts: int) -> list[HfPiece]:
    experts = sorted(pieces_by_expert)
    assert experts == list(range(experts[0], experts[-1] + 1)), f"{key}: this rank's experts are not contiguous"
    first = pieces_by_expert[experts[0]]
    assert all(
        len(pieces) == len(first)
        and all(
            p.offsets == f.offsets and p.local_elements.shape == f.local_elements.shape
            for p, f in zip(pieces, first, strict=True)
        )
        for pieces in pieces_by_expert.values()
    ), f"{key}: this rank holds different rectangles of different experts"
    return [
        HfPiece(
            key,
            (num_experts, *piece.global_shape),
            (experts[0], *piece.offsets),
            torch.stack([pieces_by_expert[e][j].local_elements for e in experts]),
        )
        for j, piece in enumerate(first)
    ]


def probe_bridge_row_mapping(
    model: Sequence[torch.nn.Module],
    params: Sequence[torch.nn.Parameter],
    export_adapter: Callable[[], Iterable[tuple[str, torch.Tensor]]],
    targets: HfLoraTargets,
    map_name: Callable[[str], HfKey] = map_peft_keys_to_hf_keys,
    unexported: torch.Tensor | None = None,
) -> RowMapping:
    """Collective: run ``export_adapter`` (a backend's HF-named export, names mapped by ``map_name``) with
    index-valued stand-ins registered in ``model`` in place of ``params``.

    ``unexported``: offsets of the elements the export leaves out by construction; no checkpoint holds them.
    """
    rank = dist.get_rank()
    total = sum(p.numel() for p in params)
    assert total < _RANK_STRIDE and dist.get_world_size() < 2**53 // _RANK_STRIDE, "the probe encoding overflows"
    pieces, a_sources, modules = [], {}, set()
    per_expert: dict[str, dict[int, list[HfPiece]]] = {}
    num_experts: dict[str, int] = {}

    def decode(name: str, tensor: torch.Tensor) -> None:
        assert tensor.dtype == torch.float64, f"{name}: the export changed the probe's dtype to {tensor.dtype}"
        codes = tensor.long()
        hf_key = map_name(name)
        module, factor = hf_key.key.rsplit(".", 1)
        components = fused_components(module, targets)
        if factor == "lora_A":
            # one Megatron A exported under several names is one shared A
            source = (codes.min().item(), codes.max().item(), codes.sum().item(), tuple(codes.shape))
            a_sources.setdefault(source, []).append(module if components is None else components)
        if hf_key.expert_index is not None:
            num_experts[hf_key.key] = max(num_experts.get(hf_key.key, 0), hf_key.expert_index + 1)
            if expert_pieces := _pieces_of(hf_key.key, codes, rank):
                per_expert.setdefault(hf_key.key, {})[hf_key.expert_index] = expert_pieces
            modules.add(module)
        elif components is None:
            pieces.extend(_pieces_of(hf_key.key, codes, rank))
            modules.add(module)
        elif factor == "lora_A":
            pieces.extend(_pieces_of(f"{components[0]}.lora_A", codes, rank))
            modules.update(components)
        else:
            for component, part in zip(components, split_fused_lora_b(module, codes, targets), strict=True):
                pieces.extend(_pieces_of(f"{component}.lora_B", part, rank))

    stand_ins, start = [], 0
    for param in params:
        codes = torch.arange(start, start + param.numel(), dtype=torch.float64, device=param.device)
        stand_ins.append(torch.nn.Parameter((codes + rank * _RANK_STRIDE).view(param.shape), requires_grad=False))
        start += param.numel()
    error = None
    try:
        with _registered_in_place_of(model, params, stand_ins):
            # the export is collective: keep consuming it after a rank-local error so no rank is left inside it
            for name, tensor in export_adapter():
                if error is None:
                    try:
                        decode(name, tensor)
                    except Exception as exc:
                        error = exc
    except Exception as exc:
        # an export failure outside Bridge's collectives goes through the rank agreement, so the other ranks fail
        # instead of waiting; a rank failing mid-collective still leaves the others hanging in Bridge's NCCL
        error = exc

    # every expert's shared A yields the same group
    groups = set()
    if error is None:
        try:
            for members in a_sources.values():
                flat = [m for member in members for m in ((member,) if isinstance(member, str) else member)]
                if len(flat) > 1:
                    groups.add(tuple(sorted(flat)))
            pieces += [
                piece
                for key, by_expert in per_expert.items()
                for piece in _stack_experts(key, by_expert, num_experts[key])
            ]
            # a group's A pieces are copies of one parameter; the library stores it once, under the group's first
            # member, which a fused module's A need not have been decoded under
            for group in groups:
                a_keys = {f"{member}.lora_A" for member in group}
                held = [p.key for p in pieces if p.key in a_keys]
                assert held, f"{group}: this rank holds no piece of the shared A"
                pieces = [
                    replace(p, key=f"{group[0]}.lora_A") if p.key == held[0] else p
                    for p in pieces
                    if p.key not in a_keys or p.key == held[0]
                ]
            covered = torch.zeros(total, dtype=torch.bool)
            for piece in pieces:
                covered[piece.local_elements.flatten()] = True
            if unexported is not None:
                assert not covered[unexported].any(), "the export reaches elements declared unexported"
                covered[unexported] = True
            assert covered.all(), f"the export does not reach {int((~covered).sum())} of this rank's elements"
        except Exception as exc:
            error = exc
    raise_if_any_rank_failed(error)

    # pipeline stages export different layers; every rank needs the whole parametrization
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (sorted(groups), sorted(modules)), group=gloo_group())
    return RowMapping(
        pieces,
        tuple(sorted({g for rank_groups, _ in gathered for g in rank_groups})),
        tuple(sorted({m for _, rank_modules in gathered for m in rank_modules})),
    )


@contextmanager
def _registered_in_place_of(
    model: Sequence[torch.nn.Module], params: Sequence[torch.nn.Parameter], stand_ins: Sequence[torch.nn.Parameter]
) -> Iterator[None]:
    """Setting a parameter's ``.data`` to another dtype replaces its grad accumulator, and with it the DDP hook
    that fills ``main_grad``; the probe registers stand-ins instead and leaves the parameters alone."""
    stand_in_by_param_id = {id(param): stand_in for param, stand_in in zip(params, stand_ins, strict=True)}
    registrations = [
        (module, name, param)
        for chunk in model
        for module in chunk.modules()
        for name, param in module._parameters.items()
        if param is not None and id(param) in stand_in_by_param_id
    ]
    assert {
        id(param) for _, _, param in registrations
    } == stand_in_by_param_id.keys(), "a probed parameter is unregistered"
    for module, name, param in registrations:
        module._parameters[name] = stand_in_by_param_id[id(param)]
    try:
        yield
    finally:
        for module, name, param in registrations:
            module._parameters[name] = param


class BridgeAdapterSlots(ProbedAdapterSlots):
    """Checkpoint members of Bridge LoRA; single LoRA is slot 0 of one, multi-LoRA slots are its MultiLoRALinears'.

    Multi-LoRA slots are padded to ``max_rank``; a slot's HF tensors have its real rank.
    """

    bf16_adapter_key_style: Literal["peft", "sampler"] = "peft"

    def __init__(self, model, iterator, targets: HfLoraTargets, *, multi_lora: bool):
        """Collective: probes the row mapping now, while nothing else touches the parameters."""
        self._model, self._iterator, self.targets, self._multi_lora = model, iterator, targets, multi_lora
        self._specs: dict[tuple[int, int | None, frozenset[int] | None], list[LocalShardSpec]] = {}
        # every slot has the same layout
        self.row_mapping = probe_bridge_row_mapping(
            model, self.slot_parameters(0), self._export_probe, targets, self._map_name, self._unexported_elements()
        )

    def _map_name(self, name: str) -> HfKey:
        return map_peft_keys_to_hf_keys(name)

    def _unexported_elements(self) -> torch.Tensor | None:
        # Bridge exports every adapter element
        return None

    def slot_parameters(self, slot: int) -> list[torch.nn.Parameter]:
        if self._multi_lora:
            from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

            return adapter_slot_parameters(self._model, slot)
        assert slot == 0, "single LoRA has only slot 0"
        return [param for params in adapter_named_parameters(self._model) for param in params.values()]

    def slot_spec(self, slot: int) -> AdapterSpec | None:
        if not self._multi_lora:
            return None
        from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

        module = next(m for chunk in self._model for m in chunk.modules() if isinstance(m, MultiLoRALinear))
        return AdapterSpec(slot=slot, rank=int(module.rank_values[slot]), alpha=float(module.alpha_values[slot]))

    def _export_probe(self):
        if not self._multi_lora:
            return self._iterator.iter_current_adapter()
        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        def exported():
            with expose_adapter_slot(self._model, 0):
                yield from self._iterator.iter_current_adapter()

        return exported()


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
    # .get: a field newer than the Bridge that wrote the config is absent and takes the dataclass default, which is
    # Bridge's behaviour before the field (experts_shared_outer_loras, share_expert_adapters, normalize_moe_lora)
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


def bridge_base_model_candidates(config: dict) -> tuple[str | None, str | None]:
    """Bridge PEFT loads its base from pretrained_checkpoint. When that is no HF dir here (torch_dist, or another
    machine's path), the HF template it was converted from stands in, as Bridge's _resolve_hf_source picks it; that
    stand-in misses a base fine-tuned in Megatron."""
    checkpoint = config["checkpoint"]
    # hf_source_path and hf_model_id are newer than some Bridge configs; Bridge takes the first non-empty one even
    # when it is no HF dir here, so a later one would compare a model Bridge never used
    template = (
        checkpoint.get("hf_source_path")
        or config["model"].get("hf_model_id")
        or config["tokenizer"]["tokenizer_model"]
    )
    return checkpoint["pretrained_checkpoint"], template


class MegatronBridgeCheckpointLoader:
    """A Megatron-Bridge PEFT ``iter_N/`` (torch_dist), read straight into this run's model and DistributedOptimizer
    under any TP / PP / EP / DP layout. The source's iteration, scheduler and RNG are not resumed."""

    def __init__(
        self, model, optimizer, members: AdapterCheckpointMembers, run: AdapterParametrization, load_optimizer: bool
    ):
        self._model, self._optimizer, self._members = model, optimizer, members
        self._run, self._load_optimizer = run, load_optimizer

    def __call__(self, path: Path) -> AdapterParametrization:
        chunks = unwrap_model(self._model)
        names = ["model"] if len(chunks) == 1 else [f"model{i}" for i in range(len(chunks))]
        config, model_state_dicts, adapter, error = None, None, None, None
        try:
            config = yaml.safe_load((path / "run_config.yaml").read_text())
            assert config["checkpoint"]["ckpt_format"] == "torch_dist", f"{path} is not a torch_dist checkpoint"
            assert_bridge_run_config_matches(
                config, self._run, params_dtype=chunks[0].config.params_dtype, load_optimizer=self._load_optimizer
            )
            assert_same_base_model(str(path), bridge_base_model_candidates(config), self._run.base_model)
            metadata = dict(dist_checkpointing.load_content_metadata(str(path)) or {})
            if self._load_optimizer:
                assert all(
                    isinstance(sub, DistributedOptimizer) for sub in self._optimizer.chained_optimizers
                ), "a Megatron-Bridge source's Adam state loads only into the DistributedOptimizer; pass --no-load-optim"
                sharding = metadata.get("distrib_optim_sharding_type")
                assert sharding is not None, (
                    f"{path} records no optimizer sharding type (written by an older Megatron), so its Adam state "
                    "cannot be resharded; pass --no-load-optim"
                )
                assert sharding in DistributedOptimizer.checkpoint_fully_reshardable_formats, (
                    f"{path}'s optimizer state is {sharding}, which only loads under its saving layout; re-save it "
                    "from Megatron-Bridge with dist_ckpt_optim_fully_reshardable=True, or pass --no-load-optim"
                )
            metadata["dp_cp_group"] = get_parallel_state().intra_dp_cp.group
            model_state_dicts = [chunk.sharded_state_dict(metadata=metadata) for chunk in chunks]
            adapter = [
                {k: v for k, v in sd.items() if ".adapter." in k and "_extra_state" not in k}
                for sd in model_state_dicts
            ]
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)

        # pipeline stages hold different layers; the checkpoint holds them all
        expected = [None] * dist.get_world_size()
        dist.all_gather_object(expected, {t.key for sd in adapter for t in nested_values(sd)}, group=gloo_group())
        expected = set().union(*expected)
        requested = dict(zip(names, adapter, strict=True))
        try:
            saved = {
                k
                for k in dcp.FileSystemReader(path).read_metadata().state_dict_metadata
                if ".adapter." in k and "_extra_state" not in k and not OPTIMIZER_KEY.match(k)
            }
            assert (
                expected == saved
            ), f"{path} holds other adapters than this run's model, e.g. {sorted(expected ^ saved)[:3]}"
            if self._load_optimizer:
                requested["optimizer"] = self._optimizer.sharded_state_dict(
                    dict(zip(names, model_state_dicts, strict=True)), is_loading=True, metadata=metadata
                )
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)

        try:
            # a rank-local failure inside load's own collectives still hangs; one after them is agreed below
            loaded = dist_checkpointing.load(requested, str(path))
            for name, chunk in zip(names, chunks, strict=True):
                result = chunk.load_state_dict(loaded[name], strict=False)
                assert not result.unexpected_keys, f"unexpected adapter keys {result.unexpected_keys[:3]}"
            if self._load_optimizer:
                # the run owns the schedule: only state and step come from the source, not its lr or weight decay
                run_hyperparameters = [
                    {k: v for k, v in g.items() if k not in ("params", "step")} for g in self._optimizer.param_groups
                ]
                try:
                    self._optimizer.load_state_dict(loaded["optimizer"])
                except KeyError as exc:
                    # the DistributedOptimizer matches param groups by (wd_mult, lr_mult, is_expert_parallel,
                    # is_decoupled_lr) and raises a bare KeyError of the missing key
                    raise ValueError(
                        f"{path} has no optimizer param group {exc} of this run's; pass --no-load-optim"
                    ) from exc
                for group, hyperparameters in zip(self._optimizer.param_groups, run_hyperparameters, strict=True):
                    group.update(hyperparameters)
            else:
                self._optimizer.reload_model_params()
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        # collective; asserts each bf16 adapter parameter is its fp32 master
        self._members.export_adapter_state(0)
        return replace(self._run, lora_alpha=float(config["peft"]["alpha"]))
