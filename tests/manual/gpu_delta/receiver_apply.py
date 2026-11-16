"""The receiver's apply path driven by hand, for the manual CUDA checks.

``apply_prepared_masks`` runs one prepared grouped batch of matrix masks, under every decoder failure first. It uses
``PreparedDelta``'s own batch step and the derived images' own refresh without a publication or nvCOMP, which have
their own checks.
"""

from types import SimpleNamespace

import torch


@torch.no_grad()
def apply_prepared_masks(bindings, masks, derived=()):
    """Exercise the prepared grouped path without autograd, as PreparedDelta.apply does; nvCOMP has its own oracle."""
    from miles.backends.sglang_utils.gpu_delta.receiver import PreparedDelta, _PreparedBatch
    from miles.utils.gpu_delta.apply import plan_apply, prepare_status_check

    outputs, size = [], 0
    for binding, mask in zip(bindings, masks, strict=False):
        offset = (size + 15) // 16 * 16
        outputs.append((binding, offset, mask.numel()))
        size = offset + mask.numel()
    payload = torch.zeros(size, dtype=torch.uint8, device="cuda")
    for (_, offset, length), mask in zip(outputs, masks, strict=False):
        payload[offset : offset + length].copy_(mask.reshape(-1))
    prepared = PreparedDelta.__new__(PreparedDelta)
    prepared.error = torch.zeros(1, dtype=torch.int32, device="cuda")
    decoded = torch.empty(size, dtype=torch.uint8, device="cuda")
    outputs = [(view, offset, size) for binding, offset, size in outputs for view in binding.apply_bindings()]
    group = plan_apply(outputs)
    saved = [(value, value.clone()) for binding in bindings for value in binding.storage]
    saved.extend((image.destination, image.destination.clone()) for image in derived)

    def apply_derived():
        for image in derived:
            image.refresh(prepared.error == 0)

    apply = None
    if group is not None:
        group.prepare_metadata(decoded.device.index)
        group.bind(decoded)
        pointers = torch.tensor(
            list(group.pointer_rows(decoded.data_ptr())),
            dtype=torch.int64,
            device="cuda",
        )
        metadata = torch.tensor(group.metadata_rows, dtype=torch.int64, device="cuda")
        apply = group.bind_launch(pointers, metadata, prepared.error)
    # Compilation and module loading must not execute against live weights.
    for value, before in saved:
        torch.testing.assert_close(
            value.contiguous().view(torch.uint8),
            before.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
    decoder = SimpleNamespace(
        enqueue=lambda: decoded[:size].copy_(payload),
        statuses=torch.zeros(1, dtype=torch.int32, device="cuda"),
        actual_sizes=torch.tensor([size], device="cuda"),
        expected_sizes=torch.tensor([size], device="cuda"),
    )
    batch = _PreparedBatch(
        decoder,
        apply,
        [],
        prepare_status_check(decoder, prepared.error),
    )
    # Prior errors, a new decoder error and a wrong decoded size must suppress
    # every affine mapping, including padded scale destinations.
    for prior, status, actual in ((1, 0, size), (0, 1, size), (0, 0, size - 1)):
        prepared.error.fill_(prior)
        decoder.statuses.fill_(status)
        decoder.actual_sizes.fill_(actual)
        decoder.enqueue()
        prepared._apply_batch(batch)
        apply_derived()
        assert prepared.error.item() == 1
        for value, before in saved:
            torch.testing.assert_close(
                value.contiguous().view(torch.uint8),
                before.contiguous().view(torch.uint8),
                rtol=0,
                atol=0,
            )
    prepared.error.zero_()
    decoder.statuses.zero_()
    decoder.actual_sizes.fill_(size)
    decoder.enqueue()
    prepared._apply_batch(batch)
    apply_derived()
    return group
