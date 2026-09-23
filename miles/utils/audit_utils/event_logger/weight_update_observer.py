from argparse import Namespace
from collections.abc import Mapping, Sequence

import torch

from miles.backends.training_utils.weight_update.utils import hash_tensor_sha256
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import WeightUpdateTransferChecksumEvent
from miles.utils.test_utils.fault_injector.controller import fault_hook_controller


def is_observing_checksums(args: Namespace) -> bool:
    return (
        args.log_inference_engine_weight_checksums
        and is_event_logger_initialized()
        and args.update_weight_transfer_mode == "p2p"
    )


def compute_send_checksums(parameters: Mapping[str, torch.Tensor], names: Sequence[str]) -> dict[str, str]:
    assert len(names) == len(set(names)), "P2P bucket contains duplicate tensor names"
    checksums = {}
    for name in names:
        tensor = parameters[name]
        assert tensor.device.type == "cpu", "Expected registered CPU send buffers"
        checksums[name] = hash_tensor_sha256(tensor)
    return checksums


def log_transfer_checksums(
    *,
    cell_id: str,
    receiver_rank: int,
    sent_checksums: dict[str, str],
    received_checksums: dict[str, str],
) -> None:
    context = fault_hook_controller.context
    assert context is not None, "Weight update checksums observed outside a weight update"
    get_event_logger().log(
        WeightUpdateTransferChecksumEvent,
        dict(
            debug_weight_update_id=context.debug_weight_update_id,
            cell_id=cell_id,
            workers_hash=context.snapshot_cell_id_to_hashes[cell_id],
            receiver_rank=receiver_rank,
            sent_checksums=sent_checksums,
            received_checksums=received_checksums,
        ),
        print_log=False,
    )
