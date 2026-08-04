"""Opt-in module-level dtype/shape tracing of the FSDP training forward.

Off unless sglang's Dumper is enabled (``DUMPER_ENABLE=1 DUMPER_NON_INTRUSIVE_MODE=all``), in which
case every ``nn.Module`` under the actor's model gets an IO-recording hook and the module/param tree
is written to ``$DUMPER_DIR/module_structure.json``. Feeding those two artifacts to the
``dumper-module-report`` skill yields the per-module dtype inventory that precision alignment starts
from: which weights gather at which dtype, and where the activation dtype changes hands.

Set ``DUMPER_ENABLE_OUTPUT_FILE=0`` to keep tensors off disk — the report only needs the console log.
"""

import logging

logger = logging.getLogger(__name__)


def _dumper():
    from sglang.srt.debug_utils.dumper import dumper

    return dumper


def maybe_register_module_dumper(model) -> None:
    """Register the non-intrusive dumper on ``model`` once, and dump its structure. No-op when off."""
    dumper = _dumper()
    if not dumper.may_enable or dumper._non_intrusives:
        return
    dumper.register_non_intrusive_dumper(model)
    _dump_module_structure(model)


def maybe_dumper_step() -> None:
    """Advance the dumper's step counter at the end of a forward. No-op when off."""
    dumper = _dumper()
    if dumper.may_enable:
        dumper.step()


def _dump_module_structure(model, out_path=None) -> None:
    """Write the module/param/buffer tree (shape+dtype) as JSON.

    DTensor-safe: under FSDP2 a param's ``.shape`` is already the global shape, so the tree describes
    the unsharded model even though every rank only holds a shard.
    """
    import json
    from pathlib import Path

    def describe(tensor):
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "class": type(tensor).__name__}

    info = {
        name or "<root>": {
            "class": type(module).__module__ + "." + type(module).__name__,
            "params": {pn: describe(p) for pn, p in module.named_parameters(recurse=False)},
            "buffers": {bn: describe(b) for bn, b in module.named_buffers(recurse=False)},
            "children": [cn for cn, _ in module.named_children()],
        }
        for name, module in model.named_modules()
    }
    out = Path(out_path) if out_path else Path(_dumper()._config.dir) / "module_structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1))
    logger.info(f"[debug_dump] wrote {out} ({len(info)} modules)")
