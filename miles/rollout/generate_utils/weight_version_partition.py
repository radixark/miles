# Namespacing generation requests by the weight version seen when a session
# started keeps radix-cache prefix reuse within one weight generation: pause
# modes that skip engine-side cache flushes (in_place) would otherwise let KV
# computed under old weights serve unrelated new requests indefinitely.

from miles.utils.types import Sample

WEIGHT_VERSION_EXTRA_KEY_METADATA_KEY: str = "weight_version_extra_key"

_latest_weight_version: int | None = None


def observe_weight_version(meta_info: dict) -> None:
    global _latest_weight_version

    try:
        version = int(meta_info.get("weight_version"))
    except (TypeError, ValueError):
        return
    _latest_weight_version = version if _latest_weight_version is None else max(_latest_weight_version, version)


def latest_weight_version() -> int | None:
    return _latest_weight_version


def lock_weight_version_extra_key(sample: Sample) -> str:
    if (extra_key := sample.metadata.get(WEIGHT_VERSION_EXTRA_KEY_METADATA_KEY)) is None:
        extra_key = format_weight_version_extra_key(_latest_weight_version)
        sample.metadata[WEIGHT_VERSION_EXTRA_KEY_METADATA_KEY] = extra_key
    return extra_key


def format_weight_version_extra_key(weight_version: int | None) -> str:
    return f"weight-version:{0 if weight_version is None else weight_version}"
