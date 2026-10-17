from collections import defaultdict
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotGeneratedValue,
    ConfigSnapshotGeneration,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.process_identity import ProcessIdentity, TrainProcessIdentity

_RANK = "$RANK"


def normalize_record(
    record: ConfigSnapshotRecord,
    *,
    generated_values: list[ConfigSnapshotGeneratedValue] | None = None,
) -> JsonValue:
    context = record.context
    config = _simple_replace(record.config, src_text=context.run_uuid, dst_text="$RUN_UUID")
    config = _normalize_generated_values(
        config, values=record.generated_values if generated_values is None else generated_values
    )
    for fields in _config_objects(config):
        if isinstance(fields.get("wandb_run_id"), str):
            fields["wandb_run_id"] = "$WANDB_RUN_ID_0000"
    if isinstance(context.source, TrainProcessIdentity):
        if not isinstance(config, dict) or not isinstance(args := config.get("args"), dict):
            raise ValueError("Training snapshots require a config.args object")
        if "rank" in args:
            assert type(args["rank"]) is int and args["rank"] >= 0, f"Unexpected args.rank: {args['rank']!r}"
            args["rank"] = _RANK
        if "backend" in args:
            backend = args["backend"]
            assert isinstance(backend, dict), "Training snapshots require a config.args.backend object"
            if "rank" in backend:
                assert (
                    type(backend["rank"]) is int and backend["rank"] >= 0
                ), f"Unexpected args.backend.rank: {backend['rank']!r}"
                backend["rank"] = _RANK
    return config


_PATH_FIELDS = frozenset(
    {
        "save",
        "load",
        "requested_load",
        "critic_save",
        "critic_load",
        "ref_load",
        "dump_details",
        "save_debug_event_data",
        "save_debug_train_data",
        "save_debug_trajectory_data",
        "save_debug_rollout_data",
        "load_debug_rollout_data",
        "ci_save_grad_norm",
        "te_precision_config_file",
        "eval_hf_dir",
    }
)


_CONFIG_PATHS = (("args",), ("backend",), ("raw_megatron", "base_args"))


def _normalize_generated_values(config: JsonValue, *, values: list[ConfigSnapshotGeneratedValue]) -> JsonValue:
    result = deepcopy(config)
    if isinstance(result, dict) and isinstance(args := result.get("args"), dict):
        _normalize_endpoints(args, values=values)
    for fields in _config_objects(result):
        fields.update(
            {
                key: _normalize_field(value, key=key, values=values, config=fields)
                for key, value in fields.items()
                if isinstance(value, str) and (key in _PATH_FIELDS or key == "wandb_group")
            }
        )
    return result


def _normalize_endpoints(fields: dict[str, JsonValue], *, values: list[ConfigSnapshotGeneratedValue]) -> None:
    hosts = {entry.value for entry in values if entry.kind == "host"}
    ports = {entry.value for entry in values if entry.kind == "port"}
    external_hosts = {entry.value for entry in values if entry.kind == "external_host"}
    if not (hosts or ports or external_hosts):
        return

    if isinstance(host := fields.get("sglang_router_ip"), str) and host in hosts:
        fields["sglang_router_ip"] = "$HOST"
    if type(port := fields.get("sglang_router_port")) is int and str(port) in ports:
        fields["sglang_router_port"] = "$PORT"
    if isinstance(routers := fields.get("sglang_model_routers"), dict):
        for model, value in routers.items():
            if not isinstance(value, dict) or not isinstance(parts := value.get("$tuple"), list) or len(parts) != 2:
                raise ValueError(f"Invalid router endpoint snapshot: {model}")
            host, port = parts
            value["$tuple"] = [
                "$HOST" if isinstance(host, str) and host in hosts else host,
                "$PORT" if type(port) is int and str(port) in ports else port,
            ]
    if isinstance(instances := fields.get("session_server_instances"), list):
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            for field in ("addr", "external_addr"):
                if not isinstance(value := instance.get(field), str):
                    raise ValueError(f"Invalid session endpoint snapshot: {instance}")
                host, separator, port = value.rpartition(":")
                if not separator or not port.isdecimal():
                    raise ValueError(f"Invalid session endpoint address: {value}")
                host = "$HOST" if host in (external_hosts if field == "external_addr" else hosts) else host
                port = "$PORT" if str(int(port)) in ports else port
                instance[field] = f"{host}:{port}"


def _config_objects(config: JsonValue) -> Iterator[dict[str, JsonValue]]:
    if not isinstance(config, dict):
        return
    yield config
    for path in _CONFIG_PATHS:
        child: JsonValue = config
        for key in path:
            child = child.get(key) if isinstance(child, dict) else None
        yield from _config_objects(child)


def _normalize_field(
    value: str, *, key: str, values: list[ConfigSnapshotGeneratedValue], config: dict[str, JsonValue]
) -> str:
    for entry in values:
        token = f"${entry.kind.upper()}_{entry.name}"
        if key == "wandb_group":
            if entry.kind == "ci_commit_name" and value.endswith(f"_{entry.value}"):
                value = value[: -len(entry.value)] + token
            elif entry.kind == "run_id":
                value = "_".join(token if part == entry.value else part for part in value.split("_"))
        elif entry.kind == "run_id":
            if key == "critic_save" and value.rsplit("/", 1)[-1] == f"{entry.value}_critic":
                save = config.get("save")
                if not isinstance(save, str) and isinstance(backend := config.get("backend"), dict):
                    save = backend.get("save")
                if isinstance(save, str) and value == save.rstrip("/") + "_critic":
                    value = value[: -len(entry.value + "_critic")] + token + "_critic"
            value = "/".join(token if part == entry.value else part for part in value.split("/"))
        elif entry.kind == "temporary_directory" and (value == entry.value or value.startswith(entry.value + "/")):
            value = str(Path(entry.value).parent / token) + value[len(entry.value) :]
    return value


def normalized_source_name(source: ProcessIdentity) -> str:
    return source.to_cell_name() if isinstance(source, TrainProcessIdentity) else source.to_name()


def collect_generated_values(
    records: list[ConfigSnapshotRecord],
) -> dict[ConfigSnapshotGeneration, list[ConfigSnapshotGeneratedValue]]:
    by_generation: dict[ConfigSnapshotGeneration, list[ConfigSnapshotRecord]] = defaultdict(list)
    for record in records:
        by_generation[record.context.generation].append(record)

    result = {}
    for generation, group in by_generation.items():
        candidates = {entry for record in group for entry in record.generated_values}
        observed: dict[tuple[str, str], ConfigSnapshotGeneratedValue] = {}
        for entry in sorted(candidates, key=lambda entry: (entry.kind, entry.name, entry.value)):
            if any(_normalize_generated_values(record.config, values=[entry]) != record.config for record in group):
                key = (entry.kind, entry.name)
                if observed.setdefault(key, entry) != entry:
                    raise ValueError(f"Conflicting generated snapshot value for {generation}/{key}")
        result[generation] = list(observed.values())
    return result


def _simple_replace(value: JsonValue, *, src_text: str, dst_text: str) -> JsonValue:
    if isinstance(value, str):
        return value.replace(src_text, dst_text)
    if isinstance(value, dict):
        return {key: _simple_replace(item, src_text=src_text, dst_text=dst_text) for key, item in value.items()}
    if isinstance(value, list):
        return [_simple_replace(item, src_text=src_text, dst_text=dst_text) for item in value]
    return value
