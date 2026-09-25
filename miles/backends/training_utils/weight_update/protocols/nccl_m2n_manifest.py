"""Pure topology, ownership, and atomic routing for NCCL M2N weight updates."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

_SCHEMA_VERSION = 1
_FP8_BLOCK_SIZE = (128, 128)
_FP8_MANIFEST_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": list(_FP8_BLOCK_SIZE),
    "weight_dtype": "float8_e4m3fn",
    "scale_dtype": "float32",
    "scale_format": "canonical",
}
_DENSE_RE = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.mlp\.linear_fc([12])\.weight$")
_EXPERT_RE = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$")


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if getattr(torch, name, None) is not dtype:
        raise ValueError(f"Unsupported NCCL M2N dtype {dtype}")
    return name


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported NCCL M2N dtype {name!r}")
    return dtype


def _tensor_bytes(shape: Sequence[int], dtype_name: str) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * torch.empty((), dtype=_dtype_from_name(dtype_name)).element_size()


def _fp8_manifest_quantization(
    quantization_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if quantization_config is None:
        return None
    if not isinstance(quantization_config, Mapping):
        raise ValueError("NCCL M2N quantization_config must be a mapping")
    try:
        json.dumps(quantization_config, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("NCCL M2N quantization_config must be JSON serializable") from exc
    if (
        quantization_config.get("quant_method") != "fp8"
        or quantization_config.get("fmt", "e4m3") != "e4m3"
        or quantization_config.get("activation_scheme") != "dynamic"
        or list(quantization_config.get("weight_block_size") or ()) != list(_FP8_BLOCK_SIZE)
    ):
        raise ValueError(
            "NCCL M2N supports only block FP8 rollout weights with "
            "quant_method='fp8', fmt='e4m3', activation_scheme='dynamic', "
            "weight_block_size=[128, 128]"
        )
    # Checkpoint scale_fmt does not determine the wire format. The sender
    # selects it to match the rollout backend, as in the broadcast path.
    scale_format = quantization_config.get("scale_format", "canonical")
    if scale_format not in ("canonical", "ue8m0_unpacked"):
        raise ValueError(f"Unsupported NCCL M2N scale format {scale_format!r}")
    return {**_FP8_MANIFEST_QUANTIZATION, "scale_format": scale_format}


def _fp8_scale_shape(
    weight_shape: Sequence[int],
    description: str,
) -> list[int]:
    shape = list(weight_shape)
    if len(shape) < 2 or any(dim % block for dim, block in zip(shape[-2:], _FP8_BLOCK_SIZE, strict=True)):
        raise ValueError(f"{description} shape {shape} is not aligned to FP8 blocks " f"{list(_FP8_BLOCK_SIZE)}")
    shape[-2] //= _FP8_BLOCK_SIZE[0]
    shape[-1] //= _FP8_BLOCK_SIZE[1]
    return shape


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _split_manifest_by_pp(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Give each PP owner a communicator-local manifest and staging namespace."""
    source_world = manifest["source_world_ranks"]
    destination_count = manifest["communicator_world_size"] - len(source_world)
    stages: dict[int, dict[str, Any]] = {}
    owners: dict[int, int] = {}
    for pp_rank in sorted({entry["pp_rank"] for entry in manifest["entries"]}):
        entries = deepcopy([entry for entry in manifest["entries"] if entry["pp_rank"] == pp_rank])
        source_ranks = sorted({rank for entry in entries for row in entry["source"]["mesh"] for rank in row})
        for rank in source_ranks:
            previous = owners.setdefault(rank, pp_rank)
            if previous != pp_rank:
                raise ValueError(f"Trainer communicator rank {rank} belongs to multiple PP stages")
        rank_map = {rank: local_rank for local_rank, rank in enumerate(source_ranks)}
        rank_map.update({len(source_world) + rank: len(source_ranks) + rank for rank in range(destination_count)})
        for entry in entries:
            for side in ("source", "destination"):
                entry[side]["mesh"] = [[rank_map[rank] for rank in row] for row in entry[side]["mesh"]]
            entry["source"]["names_by_rank"] = {
                str(rank_map[int(rank)]): names for rank, names in entry["source"]["names_by_rank"].items()
            }
        stage_world = [source_world[rank] for rank in source_ranks]
        stage = {
            "schema_version": manifest["schema_version"],
            "pp_rank": pp_rank,
            "source_world_ranks": stage_world,
            "trainer_world_to_comm_rank": {str(rank): local_rank for local_rank, rank in enumerate(stage_world)},
            "communicator_world_size": len(stage_world) + destination_count,
            "entries": entries,
        }
        if "quantization" in manifest:
            stage["quantization"] = deepcopy(manifest["quantization"])
            _validate_fp8_pairs(entries)
        stage["manifest_hash"] = _manifest_digest(stage)
        stages[pp_rank] = stage
    return stages


def _one_owner(candidates: list[dict[str, Any]], description: str) -> int:
    if len(candidates) != 1:
        ranks = sorted(item["world_rank"] for item in candidates)
        raise ValueError(f"NCCL M2N requires exactly one canonical {description}; " f"found world ranks {ranks}")
    return candidates[0]["world_rank"]


def _build_rank_layout(
    topologies: Sequence[dict[str, Any]],
    *,
    need_dense: bool,
    need_expert: bool,
) -> dict[str, Any]:
    """Select canonical dense and expert owners independently for every PP stage."""

    if not topologies:
        raise ValueError("Cannot construct NCCL M2N rank layout without trainer ranks")
    if len({item["world_rank"] for item in topologies}) != len(topologies):
        raise ValueError("Trainer topology contains duplicate world ranks")

    sizes: dict[str, int] = {}
    for field in ("pp", "tp", "cp", "dense_dp", "ep", "etp", "expert_dp", "independent_dp"):
        values = {int(item[f"{field}_size"]) for item in topologies}
        if len(values) != 1:
            raise ValueError(f"Inconsistent trainer {field} sizes: {sorted(values)}")
        sizes[field] = values.pop()
        for item in topologies:
            rank = int(item[f"{field}_rank"])
            if not 0 <= rank < sizes[field]:
                raise ValueError(
                    f"Invalid trainer {field} rank {rank} for size {sizes[field]} "
                    f"on world rank {item['world_rank']}"
                )
    if sizes["etp"] != 1:
        raise ValueError(f"NCCL M2N requires trainer ETP=1, got {sizes['etp']}")

    coordinate_views = {
        "dense": ("pp", "tp", "cp", "dense_dp", "independent_dp"),
        "expert": ("pp", "ep", "etp", "expert_dp", "independent_dp"),
    }
    for family, fields in coordinate_views.items():
        expected = 1
        for field in fields:
            expected *= sizes[field]
        coordinates = {tuple(int(item[f"{field}_rank"]) for field in fields) for item in topologies}
        if len(topologies) != expected or len(coordinates) != expected:
            raise ValueError(
                f"NCCL M2N {family} topology describes {expected} ranks across "
                f"{fields}, but received {len(topologies)} trainer ranks with "
                f"{len(coordinates)} unique coordinates"
            )

    canonical = [item for item in topologies if item["independent_dp_rank"] == 0]
    dense_world_by_pp: dict[str, list[int]] = {}
    expert_world_by_pp: dict[str, list[int]] = {}
    for pp_rank in range(sizes["pp"]):
        if need_dense:
            dense_world_by_pp[str(pp_rank)] = [
                _one_owner(
                    [
                        item
                        for item in canonical
                        if item["pp_rank"] == pp_rank
                        and item["dense_dp_rank"] == 0
                        and item["cp_rank"] == 0
                        and item["tp_rank"] == tp_rank
                    ],
                    f"dense owner for PP={pp_rank}, TP={tp_rank}",
                )
                for tp_rank in range(sizes["tp"])
            ]
        if need_expert:
            expert_world_by_pp[str(pp_rank)] = [
                _one_owner(
                    [
                        item
                        for item in canonical
                        if item["pp_rank"] == pp_rank
                        and item["expert_dp_rank"] == 0
                        and item["ep_rank"] == ep_rank
                        and item["etp_rank"] == etp_rank
                    ],
                    f"expert owner for PP={pp_rank}, EP={ep_rank}, ETP={etp_rank}",
                )
                for ep_rank in range(sizes["ep"])
                for etp_rank in range(sizes["etp"])
            ]

    ordered_world = sorted(
        {rank for meshes in (dense_world_by_pp, expert_world_by_pp) for mesh in meshes.values() for rank in mesh}
    )
    if not ordered_world:
        raise ValueError("No trainer rank owns an NCCL M2N-routable tensor")
    world_to_comm = {world_rank: comm_rank for comm_rank, world_rank in enumerate(ordered_world)}
    dense_meshes = {pp: [world_to_comm[rank] for rank in ranks] for pp, ranks in dense_world_by_pp.items()}
    expert_meshes = {pp: [world_to_comm[rank] for rank in ranks] for pp, ranks in expert_world_by_pp.items()}
    for family, meshes in (("dense", dense_meshes), ("expert", expert_meshes)):
        for pp_rank, mesh in meshes.items():
            if mesh != list(range(mesh[0], mesh[0] + len(mesh))):
                raise ValueError(
                    f"NCCL M2N {family} source mesh for PP={pp_rank} must be "
                    f"a contiguous communicator interval, got {mesh}"
                )

    return {
        "source_world_ranks": ordered_world,
        "trainer_world_to_comm_rank": {str(key): value for key, value in world_to_comm.items()},
        "dense_world_by_pp": dense_world_by_pp,
        "expert_world_by_pp": expert_world_by_pp,
        "dense_source_mesh_by_pp": dense_meshes,
        "expert_source_mesh_by_pp": expert_meshes,
        "sizes": sizes,
    }


def _local_source_spec(
    name: str,
    tensor: torch.Tensor,
) -> dict[str, Any] | None:
    dense_match = _DENSE_RE.fullmatch(name)
    expert_match = _EXPERT_RE.fullmatch(name)
    if dense_match:
        layer, projection = dense_match.groups()
        family = "dense"
        expert_id = None
    elif expert_match:
        layer, projection, expert_id = expert_match.groups()
        family = "routed_expert"
        expert_id = int(expert_id)
    else:
        return None
    partition_dim = int(getattr(tensor, "partition_dim", -1))
    partition_stride = int(getattr(tensor, "partition_stride", 1))
    if family == "dense" and projection == "1":
        # Fused SwiGLU FC1 stores each local TP shard as [gate, up]. Older
        # Megatron/TE versions do not consistently expose partition_stride=2.
        partition_dim = 0
        partition_stride = 2
    elif family == "dense" and projection == "2":
        # Match the existing residual gather workaround for TE row-parallel
        # projections that incorrectly report partition_dim=0.
        partition_dim = 1
        partition_stride = 1
    return {
        "name": name,
        "family": family,
        "layer": int(layer),
        "projection": f"fc{projection}",
        "expert_id": expert_id,
        "dtype": _dtype_name(tensor.dtype),
        "local_shape": list(tensor.shape),
        "partition_dim": partition_dim,
        "partition_stride": partition_stride,
    }


def _entry(
    *,
    name: str,
    family: str,
    pp_rank: int,
    source_names_by_rank: Mapping[int, Sequence[str]],
    dtype: str,
    global_shape: Sequence[int],
    source_mesh: Sequence[int],
    source_shard_dim: int,
    source_local_shape: Sequence[int],
    destination_mesh: Sequence[Sequence[int]],
    destination_shard_dim: int,
    destination_local_shape: Sequence[int],
    source_recipe: str,
    destination_recipe: str,
    destination_parameter: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "pp_rank": pp_rank,
        "dtype": dtype,
        "global_shape": list(global_shape),
        "source": {
            "mesh": [list(source_mesh)],
            "placements": [
                {"type": "replicate"},
                {"type": "shard", "dim": source_shard_dim},
            ],
            "local_shape": list(source_local_shape),
            "names_by_rank": {str(rank): list(names) for rank, names in sorted(source_names_by_rank.items())},
            "recipe": source_recipe,
        },
        "destination": {
            "mesh": [list(row) for row in destination_mesh],
            "placements": [
                {"type": "replicate"},
                {"type": "shard", "dim": destination_shard_dim},
            ],
            "local_shape": list(destination_local_shape),
            "parameter": destination_parameter,
            "recipe": destination_recipe,
        },
    }


def _fp8_expert_pair(weight: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if weight["family"] != "routed_expert" or weight["dtype"] != "bfloat16":
        raise ValueError(
            "NCCL M2N FP8 routes require routed-expert BF16 sources, "
            f"got family={weight['family']!r} dtype={weight['dtype']!r}"
        )
    pair_id = weight["name"]
    if not pair_id.endswith(".weight"):
        raise ValueError(f"Invalid FP8 weight name {pair_id!r}")

    source = weight["source"]
    destination = weight["destination"]
    scale = _entry(
        name=f"{pair_id.removesuffix('.weight')}.weight_scale_inv",
        family=weight["family"],
        pp_rank=weight["pp_rank"],
        source_names_by_rank={int(rank): names for rank, names in source["names_by_rank"].items()},
        dtype="float32",
        global_shape=_fp8_scale_shape(weight["global_shape"], f"{pair_id} global"),
        source_mesh=source["mesh"][0],
        source_shard_dim=source["placements"][1]["dim"],
        source_local_shape=_fp8_scale_shape(source["local_shape"], f"{pair_id} source"),
        destination_mesh=destination["mesh"],
        destination_shard_dim=destination["placements"][1]["dim"],
        destination_local_shape=_fp8_scale_shape(destination["local_shape"], f"{pair_id} destination"),
        source_recipe=f"{source['recipe']}_scale",
        destination_recipe=f"{destination['recipe']}_scale",
        destination_parameter=f"{destination['parameter']}_scale_inv",
    )
    weight["dtype"] = "float8_e4m3fn"
    weight["pair_id"] = pair_id
    weight["tensor_role"] = "weight"
    scale["pair_id"] = pair_id
    scale["tensor_role"] = "scale"
    return weight, scale


def _local_shape(
    global_shape: Sequence[int],
    shard_dim: int,
    shard_count: int,
) -> list[int]:
    shape = list(global_shape)
    if shape[shard_dim] % shard_count:
        raise ValueError(f"Shape {shape} cannot shard dimension {shard_dim} over {shard_count} ranks")
    shape[shard_dim] //= shard_count
    return shape


def _expert_destination_shard_dim(
    projection: str,
    destination_tp_size: int,
    destination_ep_size: int | None,
) -> int:
    """Return the SGLang routed-expert shard dimension for an engine.

    SGLang consumes the engine's tensor-parallel ranks either entirely as
    expert parallelism (EP=TP, MoE-TP=1) or entirely as MoE tensor
    parallelism (EP=1, MoE-TP=TP). The former shards the expert axis. The
    latter replicates experts and shards each expert's intermediate axis.

    A missing EP size preserves the original manifest-builder behavior for
    callers that do not have rollout topology available.
    """

    ep_size = destination_tp_size if destination_ep_size is None else destination_ep_size
    if ep_size <= 0 or destination_tp_size % ep_size:
        raise ValueError(
            "NCCL M2N requires rollout TP to be divisible by rollout EP; "
            f"got TP={destination_tp_size}, EP={ep_size}"
        )
    if ep_size == destination_tp_size:
        return 0
    if ep_size == 1:
        return 1 if projection == "fc1" else 2
    raise ValueError(
        "NCCL M2N routed-expert transfer currently supports rollout EP=1 "
        "or EP=TP; "
        f"got TP={destination_tp_size}, EP={ep_size}"
    )


def _entry_source_names(entry: Mapping[str, Any]) -> set[str]:
    return {name for names in entry["source"]["names_by_rank"].values() for name in names}


def _fp8_expert_module(pair_id: str) -> tuple[str, str]:
    for component in ("gate", "up", "down"):
        suffix = f".{component}_proj.weight"
        if pair_id.endswith(suffix):
            return pair_id.removesuffix(suffix), component
    raise ValueError(f"Invalid NCCL M2N FP8 expert pair ID {pair_id!r}")


def _retain_complete_fp8_modules(
    entries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    modules: dict[str, set[tuple[str, str]]] = {}
    entry_modules: list[str] = []
    for entry in entries:
        module, component = _fp8_expert_module(entry["pair_id"])
        modules.setdefault(module, set()).add((component, entry["tensor_role"]))
        entry_modules.append(module)
    complete = {
        module
        for module, members in modules.items()
        if members == {(component, role) for component in ("gate", "up", "down") for role in ("weight", "scale")}
    }
    return [entry for entry, module in zip(entries, entry_modules, strict=True) if module in complete]


def _validate_fp8_pairs(entries: Sequence[Mapping[str, Any]]) -> None:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    modules: dict[str, set[str]] = {}
    for entry in entries:
        pair_id = entry.get("pair_id")
        role = entry.get("tensor_role")
        if (
            entry.get("family") != "routed_expert"
            or not isinstance(pair_id, str)
            or role not in ("weight", "scale")
            or role in pairs.setdefault(pair_id, {})
        ):
            raise ValueError(f"Invalid NCCL M2N FP8 pair metadata for {entry['name']}")
        pairs[pair_id][role] = entry
        module, component = _fp8_expert_module(pair_id)
        modules.setdefault(module, set()).add(component)

    for pair_id, pair in pairs.items():
        if set(pair) != {"weight", "scale"}:
            raise ValueError(f"NCCL M2N FP8 pair {pair_id!r} is incomplete: {sorted(pair)}")
        weight, scale = pair["weight"], pair["scale"]
        expected_scale_name = f"{pair_id.removesuffix('.weight')}.weight_scale_inv"
        if (
            weight["name"] != pair_id
            or scale["name"] != expected_scale_name
            or weight["dtype"] != "float8_e4m3fn"
            or scale["dtype"] != "float32"
            or weight["pp_rank"] != scale["pp_rank"]
            or _entry_source_names(weight) != _entry_source_names(scale)
            or weight["source"]["names_by_rank"] != scale["source"]["names_by_rank"]
            or weight["source"]["mesh"] != scale["source"]["mesh"]
            or weight["source"]["placements"] != scale["source"]["placements"]
            or weight["destination"]["mesh"] != scale["destination"]["mesh"]
            or weight["destination"]["placements"] != scale["destination"]["placements"]
            or scale["global_shape"] != _fp8_scale_shape(weight["global_shape"], f"{pair_id} global")
            or scale["source"]["local_shape"] != _fp8_scale_shape(weight["source"]["local_shape"], f"{pair_id} source")
            or scale["destination"]["local_shape"]
            != _fp8_scale_shape(weight["destination"]["local_shape"], f"{pair_id} destination")
            or scale["source"]["recipe"] != f"{weight['source']['recipe']}_scale"
            or scale["destination"]["recipe"] != f"{weight['destination']['recipe']}_scale"
            or scale["destination"]["parameter"] != f"{weight['destination']['parameter']}_scale_inv"
        ):
            raise ValueError(f"Inconsistent NCCL M2N FP8 pair {pair_id!r}")
    for module, components in modules.items():
        if components != {"gate", "up", "down"}:
            raise ValueError(f"NCCL M2N FP8 expert module {module!r} is incomplete: " f"{sorted(components)}")


def _same_specs(specs: Sequence[dict[str, Any]], fields: Sequence[str], description: str) -> None:
    for field in fields:
        values = {json.dumps(spec[field], sort_keys=True) for spec in specs}
        if len(values) != 1:
            raise ValueError(f"Inconsistent {description} {field}: {values}")


def _build_manifest(
    trainer_payloads: Sequence[dict[str, Any]],
    engine_gpu_counts: Sequence[int],
    quantization_config: Mapping[str, Any] | None = None,
    destination_ep_size: int | None = None,
) -> dict[str, Any]:
    quantization = _fp8_manifest_quantization(quantization_config)
    if not engine_gpu_counts or any(count <= 0 for count in engine_gpu_counts):
        raise ValueError(f"NCCL M2N requires positive rollout engine GPU counts, got {engine_gpu_counts}")
    if len(set(engine_gpu_counts)) != 1:
        raise ValueError("NCCL M2N requires homogeneous rollout engine parallelism, " f"got {list(engine_gpu_counts)}")
    destination_count = sum(engine_gpu_counts)

    all_specs = [spec for payload in trainer_payloads for spec in payload["specs"]]
    need_dense = quantization is None and any(spec["family"] == "dense" for spec in all_specs)
    need_expert = any(spec["family"] == "routed_expert" for spec in all_specs)
    destination_tp_size = engine_gpu_counts[0]
    if need_expert:
        # Validate the rollout expert layout before building rank ownership.
        # Projection-specific calls below select the corresponding shard axis.
        _expert_destination_shard_dim("fc1", destination_tp_size, destination_ep_size)
    layout = _build_rank_layout(
        [payload["topology"] for payload in trainer_payloads],
        need_dense=need_dense,
        need_expert=need_expert,
    )
    payload_by_world = {payload["topology"]["world_rank"]: payload for payload in trainer_payloads}
    world_to_comm = {int(world): comm for world, comm in layout["trainer_world_to_comm_rank"].items()}
    source_count = len(layout["source_world_ranks"])
    destination_mesh: list[list[int]] = []
    cursor = source_count
    for count in engine_gpu_counts:
        destination_mesh.append(list(range(cursor, cursor + count)))
        cursor += count
    entries: list[dict[str, Any]] = []
    layer_owners: dict[int, int] = {}

    def record_layer_owner(layer: int, pp_rank: int) -> None:
        previous = layer_owners.setdefault(layer, pp_rank)
        if previous != pp_rank:
            raise ValueError(f"Global decoder layer {layer} is reported by PP stages " f"{previous} and {pp_rank}")

    for pp_rank_text, dense_world in layout["dense_world_by_pp"].items():
        pp_rank = int(pp_rank_text)
        source_mesh = layout["dense_source_mesh_by_pp"][pp_rank_text]
        specs_by_world = {
            world: {
                (spec["layer"], spec["projection"]): spec
                for spec in payload_by_world[world]["specs"]
                if spec["family"] == "dense"
            }
            for world in dense_world
        }
        dense_keys = set.intersection(*(set(specs) for specs in specs_by_world.values()))
        if any(set(specs) != dense_keys for specs in specs_by_world.values()):
            raise ValueError(f"Dense FFN source specs differ across selected TP ranks for PP={pp_rank}")
        for layer, projection in sorted(dense_keys):
            record_layer_owner(layer, pp_rank)
            specs = [specs_by_world[world][(layer, projection)] for world in dense_world]
            _same_specs(
                specs,
                (
                    "name",
                    "dtype",
                    "local_shape",
                    "partition_dim",
                    "partition_stride",
                ),
                f"dense layer {layer} {projection}",
            )
            spec = specs[0]
            rows, columns = spec["local_shape"]
            source_names_by_rank = {
                world_to_comm[world]: [specs_by_world[world][(layer, projection)]["name"]] for world in dense_world
            }
            if projection == "fc1":
                if rows % 2 or spec["partition_dim"] != 0 or spec["partition_stride"] != 2:
                    raise ValueError(
                        f"{spec['name']} is not a supported fused gate/up TP shard: "
                        f"shape={spec['local_shape']}, partition_dim={spec['partition_dim']}, "
                        f"partition_stride={spec['partition_stride']}"
                    )
                local_rows = rows // 2
                global_shape = [local_rows * len(dense_world), columns]
                if global_shape[0] % destination_tp_size:
                    raise ValueError(f"{spec['name']} cannot shard evenly over rollout " f"TP={destination_tp_size}")
                for component, index in (("gate", 0), ("up", 1)):
                    entries.append(
                        _entry(
                            name=f"model.layers.{layer}.mlp.{component}_proj.weight",
                            family="dense",
                            pp_rank=pp_rank,
                            source_names_by_rank=source_names_by_rank,
                            dtype=spec["dtype"],
                            global_shape=global_shape,
                            source_mesh=source_mesh,
                            source_shard_dim=0,
                            source_local_shape=[local_rows, columns],
                            destination_mesh=destination_mesh,
                            destination_shard_dim=0,
                            destination_local_shape=_local_shape(global_shape, 0, destination_tp_size),
                            source_recipe=f"dense_fc1_{index}",
                            destination_recipe=f"dense_{component}",
                            destination_parameter=f"model.layers.{layer}.mlp.gate_up_proj.weight",
                        )
                    )
            else:
                if spec["partition_stride"] != 1 or spec["partition_dim"] != 1:
                    raise ValueError(
                        f"{spec['name']} is not a supported row-parallel down projection: "
                        f"partition_dim={spec['partition_dim']}, partition_stride={spec['partition_stride']}"
                    )
                global_shape = [rows, columns * len(dense_world)]
                if global_shape[1] % destination_tp_size:
                    raise ValueError(f"{spec['name']} cannot shard evenly over rollout " f"TP={destination_tp_size}")
                entries.append(
                    _entry(
                        name=f"model.layers.{layer}.mlp.down_proj.weight",
                        family="dense",
                        pp_rank=pp_rank,
                        source_names_by_rank=source_names_by_rank,
                        dtype=spec["dtype"],
                        global_shape=global_shape,
                        source_mesh=source_mesh,
                        source_shard_dim=1,
                        source_local_shape=[rows, columns],
                        destination_mesh=destination_mesh,
                        destination_shard_dim=1,
                        destination_local_shape=_local_shape(global_shape, 1, destination_tp_size),
                        source_recipe="dense_fc2",
                        destination_recipe="dense_down",
                        destination_parameter=f"model.layers.{layer}.mlp.down_proj.weight",
                    )
                )

    for pp_rank_text, expert_world in layout["expert_world_by_pp"].items():
        pp_rank = int(pp_rank_text)
        source_mesh = layout["expert_source_mesh_by_pp"][pp_rank_text]
        specs_by_world: dict[int, dict[tuple[int, str], list[dict[str, Any]]]] = {}
        for world in expert_world:
            grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
            for spec in payload_by_world[world]["specs"]:
                if spec["family"] == "routed_expert":
                    grouped.setdefault((spec["layer"], spec["projection"]), []).append(spec)
            for specs in grouped.values():
                specs.sort(key=lambda item: item["expert_id"])
            specs_by_world[world] = grouped
        expert_keys = set.intersection(*(set(specs) for specs in specs_by_world.values()))
        if any(set(specs) != expert_keys for specs in specs_by_world.values()):
            raise ValueError(f"Expert FFN source specs differ across selected EP ranks for PP={pp_rank}")
        for layer, projection in sorted(expert_keys):
            record_layer_owner(layer, pp_rank)
            per_world = [specs_by_world[world][(layer, projection)] for world in expert_world]
            counts = {len(specs) for specs in per_world}
            if len(counts) != 1 or not counts:
                raise ValueError(f"Uneven local expert counts for layer {layer} " f"{projection}: {counts}")
            local_experts = counts.pop()
            for shard, specs in enumerate(per_world):
                actual_ids = [spec["expert_id"] for spec in specs]
                expected_ids = list(
                    range(
                        shard * local_experts,
                        (shard + 1) * local_experts,
                    )
                )
                if actual_ids != expected_ids:
                    raise ValueError(
                        f"Layer {layer} {projection} source shard {shard} must "
                        f"own expert IDs {expected_ids}, got {actual_ids}"
                    )
            flat_specs = [spec for specs in per_world for spec in specs]
            expert_ids = sorted(spec["expert_id"] for spec in flat_specs)
            if expert_ids != list(range(len(expert_ids))):
                raise ValueError(
                    f"Layer {layer} {projection} expert IDs must be complete and contiguous, got {expert_ids}"
                )
            _same_specs(
                flat_specs,
                (
                    "dtype",
                    "local_shape",
                    "partition_dim",
                    "partition_stride",
                ),
                f"expert layer {layer} {projection}",
            )
            spec = flat_specs[0]
            rows, columns = spec["local_shape"]
            source_names_by_rank = {
                world_to_comm[world]: [item["name"] for item in specs_by_world[world][(layer, projection)]]
                for world in expert_world
            }
            num_experts = local_experts * len(expert_world)
            destination_shard_dim = _expert_destination_shard_dim(
                projection,
                destination_tp_size,
                destination_ep_size,
            )
            if projection == "fc1":
                if rows % 2:
                    raise ValueError(f"Expert fused gate/up rows must be even for {spec['name']}")
                intermediate = rows // 2
                global_shape = [num_experts, intermediate, columns]
                for component, index in (("gate", 0), ("up", 1)):
                    weight = _entry(
                        name=f"model.layers.{layer}.mlp.experts.{component}_proj.weight",
                        family="routed_expert",
                        pp_rank=pp_rank,
                        source_names_by_rank=source_names_by_rank,
                        dtype=spec["dtype"],
                        global_shape=global_shape,
                        source_mesh=source_mesh,
                        source_shard_dim=0,
                        source_local_shape=[
                            local_experts,
                            intermediate,
                            columns,
                        ],
                        destination_mesh=destination_mesh,
                        destination_shard_dim=destination_shard_dim,
                        destination_local_shape=_local_shape(
                            global_shape,
                            destination_shard_dim,
                            destination_tp_size,
                        ),
                        source_recipe=f"expert_fc1_{index}",
                        destination_recipe=f"expert_{component}",
                        destination_parameter=f"model.layers.{layer}.mlp.experts.w13_weight",
                    )
                    entries.extend(_fp8_expert_pair(weight) if quantization is not None else (weight,))
            else:
                global_shape = [num_experts, rows, columns]
                weight = _entry(
                    name=f"model.layers.{layer}.mlp.experts.down_proj.weight",
                    family="routed_expert",
                    pp_rank=pp_rank,
                    source_names_by_rank=source_names_by_rank,
                    dtype=spec["dtype"],
                    global_shape=global_shape,
                    source_mesh=source_mesh,
                    source_shard_dim=0,
                    source_local_shape=[local_experts, rows, columns],
                    destination_mesh=destination_mesh,
                    destination_shard_dim=destination_shard_dim,
                    destination_local_shape=_local_shape(
                        global_shape,
                        destination_shard_dim,
                        destination_tp_size,
                    ),
                    source_recipe="expert_fc2",
                    destination_recipe="expert_down",
                    destination_parameter=f"model.layers.{layer}.mlp.experts.w2_weight",
                )
                entries.extend(_fp8_expert_pair(weight) if quantization is not None else (weight,))

    if not entries:
        raise ValueError("The selected model has no FFN update unit supported by NCCL M2N")
    entries.sort(key=lambda item: (item["pp_rank"], item["name"]))
    all_units = {tuple(unit) for payload in trainer_payloads for unit in payload["update_units"]}
    while True:
        covered_names = {name for item in entries for name in _entry_source_names(item)}
        routed_units = sorted(unit for unit in all_units if unit and set(unit).issubset(covered_names))
        routed_names = {name for unit in routed_units for name in unit}
        retained_entries = [item for item in entries if _entry_source_names(item).issubset(routed_names)]
        if quantization is not None:
            retained_entries = _retain_complete_fp8_modules(retained_entries)
        if len(retained_entries) == len(entries):
            break
        entries = retained_entries
    if not entries:
        raise ValueError("No complete atomic FFN update unit can be routed through NCCL M2N")
    if quantization is not None:
        _validate_fp8_pairs(entries)
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "source_world_ranks": layout["source_world_ranks"],
        "trainer_world_to_comm_rank": layout["trainer_world_to_comm_rank"],
        "communicator_world_size": source_count + destination_count,
        "routed_update_units": [list(unit) for unit in routed_units],
        "entries": entries,
    }
    if quantization is not None:
        manifest["quantization"] = quantization
    manifest["manifest_hash"] = _manifest_digest(manifest)
    return manifest
