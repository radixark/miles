"""Trainer side of the replay data under ``--enable-delay-token-replay-assembly``.

Runs once per train attempt on this rank's DP shard, after the shard is selected
(``process_rollout_data``): reads the output store objects the shard's replay indexes name, then
replaces the shard's replay index columns with the columns training reads
(``miles.utils.replay_assembly``). Inline and store data take the same path; the inputs are only this
shard's samples, so a retry with another DP size reads again from scratch. Without the flag the
rollout executor already assembled them, the train dict has no replay index column, and this does
nothing.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from miles.utils.output_store_config import get_output_store_reader, output_store_read_buffer_bytes
from miles.utils.replay_assembly import assemble_replay_columns
from miles.utils.replay_index import ReplayDataError, ReplayIndex, pop_train_columns

logger = logging.getLogger(__name__)


def materialize_replay_columns(args: Any, rollout_data: dict[str, Any]) -> None:
    """Replace this shard's replay index columns with the columns training reads."""
    indexes = pop_train_columns(rollout_data)
    if indexes is None:
        return
    fetched = read_store_objects(indexes, buffer_bytes=_read_buffer_bytes(args, indexes))
    assemble_replay_columns(args, rollout_data, indexes, fetched=fetched)


def _read_buffer_bytes(args: Any, indexes: Sequence[ReplayIndex]) -> int:
    if not any(index.store_refs() for index in indexes):
        return 0
    return output_store_read_buffer_bytes(args)


# ============================== store reads ==============================


def read_store_objects(indexes: Sequence[ReplayIndex], *, buffer_bytes: int) -> dict[tuple[str, str], np.ndarray]:
    """Read every stored field the shard uses once, in chunks that fit the read buffer.

    Each chunk is copied into arrays this process owns before the next one is read, so the
    client's buffer only ever holds one chunk.
    """
    wanted = _collect_store_fields(indexes)
    if not wanted:
        return {}
    largest_key, largest = max(((key, spec) for key, (_, spec) in wanted.items()), key=lambda item: item[1].nbytes)
    if largest.nbytes > buffer_bytes:
        raise ReplayDataError(
            f"output store field {largest_key[0]}/{largest_key[1]} has {largest.nbytes} bytes, more than the "
            f"{buffer_bytes}-byte read buffer; raise --output-store-local-buffer-size to at least {largest.nbytes}"
        )

    from radix_transfer_plugins import StoreRef
    from radix_transfer_plugins.stashes.token_output import TokenOutputStash

    reader = get_output_store_reader()
    fetched: dict[tuple[str, str], np.ndarray] = {}
    num_chunks = 0
    for chunk in _chunk_by_bytes(wanted, buffer_bytes=buffer_bytes):
        num_chunks += 1
        refs = [StoreRef(backend=ref.backend, key=key, fields=tuple(specs)) for key, (ref, specs) in chunk.items()]
        try:
            stashes = reader.get_many(refs, stash_type=TokenOutputStash)
        except Exception as e:
            raise ReplayDataError(
                f"failed to read {len(refs)} output store objects (first keys {[ref.key for ref in refs[:4]]}): {e!r}"
            ) from e
        for ref, stash in zip(refs, stashes, strict=True):
            for spec in ref.fields:
                # A copy, so the stash's view of the read buffer is dropped before the next chunk.
                fetched[(ref.key, spec.name)] = np.array(getattr(stash, spec.name), copy=True)
        del stashes
    logger.info(
        f"Read {len(wanted)} output store fields of {len({key for key, _ in wanted})} objects, "
        f"{sum(spec.nbytes for _, spec in wanted.values())} bytes, in {num_chunks} chunks"
    )
    return fetched


def _collect_store_fields(indexes: Sequence[ReplayIndex]) -> dict[tuple[str, str], tuple[Any, Any]]:
    """(key, stored field) -> (ref, field spec), each object field once however many samples use it."""
    wanted = {}
    for index in indexes:
        for ref, stored in index.store_reads():
            specs = {spec.name: spec for spec in ref.fields}
            for name in stored:
                if name not in specs:
                    raise ReplayDataError(f"output store object {ref.key} has no field {name}")
                wanted.setdefault((ref.key, name), (ref, specs[name]))
    return wanted


def _chunk_by_bytes(
    wanted: Mapping[tuple[str, str], tuple[Any, Any]], *, buffer_bytes: int
) -> Iterator[dict[str, tuple[Any, list[Any]]]]:
    chunk: dict[str, tuple[Any, list[Any]]] = {}
    used = 0
    for (key, _), (ref, spec) in wanted.items():
        if chunk and used + spec.nbytes > buffer_bytes:
            yield chunk
            chunk, used = {}, 0
        chunk.setdefault(key, (ref, []))[1].append(spec)
        used += spec.nbytes
    if chunk:
        yield chunk
