from typing import Any

from .p2p_transfer_utils import RemoteWeightInfo


def _do_p2p_write_one_session(
    transfer_engine: Any,
    remote_session: RemoteWeightInfo,
    names: list[str],
    weight_memory_registry: dict[str, tuple[int, int, int]],
) -> None:
    """P2P write from shared CPU pinned buffers to a single remote session.

    Used by the parallelized submission path where each session within an
    rollout engine rank is submitted as a separate task to P2PTransferManager.
    """
    source_ptrs, source_lens = [], []
    valid_names = []

    for name in names:
        cpu_reg = weight_memory_registry.get(name)
        assert cpu_reg, f"the _weight_memory_registry of {name} failed"

        data_ptr, numel, ele_size = cpu_reg
        source_ptrs.append(data_ptr)
        source_lens.append(numel * ele_size)
        valid_names.append(name)

    if not source_ptrs:
        return

    session_id = remote_session.session_id
    target_ptrs = []
    for name in valid_names:
        if name in remote_session.weights_info:
            target_ptrs.append(remote_session.weights_info[name].address)

    assert len(target_ptrs) == len(source_ptrs), (
        f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
        f"source: {len(source_ptrs)}, target: {len(target_ptrs)}"
    )

    ret = transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
    if ret < 0:
        raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")
