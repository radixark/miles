from __future__ import annotations

from pathlib import Path

from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs


def make_script_args(**overrides: object) -> WorkerScriptArgs:
    defaults = dict(
        hf_checkpoint=Path("/fake/hf"),
        token_ids_file=Path("/tmp/tokens.json"),
    )
    defaults.update(overrides)
    return WorkerScriptArgs(**defaults)  # type: ignore[arg-type]
