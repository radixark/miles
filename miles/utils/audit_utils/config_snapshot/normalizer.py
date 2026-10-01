from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotRecord
from miles.utils.audit_utils.process_identity import ProcessIdentity, TrainProcessIdentity

_RANK = "$RANK"


def normalize_record(record: ConfigSnapshotRecord) -> JsonValue:
    context = record.context
    config = _simple_replace(record.config, src_text=context.run_uuid, dst_text="$RUN_UUID")
    if isinstance(context.source, TrainProcessIdentity):
        if not isinstance(config, dict) or not isinstance(args := config.get("args"), dict):
            raise ValueError("Training snapshots require a config.args object")
        if "rank" in args:
            assert type(args["rank"]) is int and args["rank"] >= 0, f"Unexpected args.rank: {args['rank']!r}"
            args["rank"] = _RANK
    return config


def normalized_source_name(source: ProcessIdentity) -> str:
    return source.to_cell_name() if isinstance(source, TrainProcessIdentity) else source.to_name()


def _simple_replace(value: JsonValue, *, src_text: str, dst_text: str) -> JsonValue:
    if isinstance(value, str):
        return value.replace(src_text, dst_text)
    if isinstance(value, dict):
        return {key: _simple_replace(item, src_text=src_text, dst_text=dst_text) for key, item in value.items()}
    if isinstance(value, list):
        return [_simple_replace(item, src_text=src_text, dst_text=dst_text) for item in value]
    return value
