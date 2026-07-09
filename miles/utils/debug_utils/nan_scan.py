"""NaN/Inf scanning for numerical debugging.

Gated on MILES_NAN_SCAN=1 (the --debug-nan-scan train arg); when disabled every
call is a no-op, so call sites need no guards. Shared state (step tag, grad-hook
dedup) lives on the module-level `nan_scanner` instance, mirroring `dumper`:

    from miles.utils.debug_utils.nan_scan import nan_scanner

    nan_scanner.scan("logits", logits)              # one stats line: nan/inf counts, max_abs
    nan_scanner.scan("logits", logits, grad=True)   # also scan its gradient in backward
    nan_scanner.scan("batch", batch)                # dict/list of tensors; also nn.Module
    nan_scanner.scan("diff", lambda: a - b)         # callable: evaluated only when enabled
    nan_scanner.scan("x", x, quiet=True)            # print only on violation
    nan_scanner.step()                              # bump the step= tag (per microbatch)

Output, with a ***NONFINITE*** suffix on violations:

    [NAN_SCAN] rank=62 step=3 logits: shape=(5120, 1, 19360) nan=0 inf=0 max_abs=28.5

Unlike the Dumper (offline dumps, cross-run comparison), this answers "where is
the first non-finite value in THIS run" inline, with no files and no second
run. Every enabled call synchronizes the GPU.
"""

import os
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

_MAX_OFFENDER_LINES = 8


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return -1


def _tensor_counts(t: Tensor) -> tuple[int, int, float | None]:
    """Return (nan_count, inf_count, max_abs_finite) for a tensor."""
    flat = t.detach().reshape(-1)
    if flat.numel() == 0:
        return 0, 0, None
    nan = int(torch.isnan(flat).sum().item())
    inf = int(torch.isinf(flat).sum().item())
    finite = flat[torch.isfinite(flat)] if (nan or inf) else flat
    max_abs = float(finite.abs().max().item()) if finite.numel() > 0 else None
    return nan, inf, max_abs


def _fmt(max_abs: float | None) -> str:
    return "none" if max_abs is None else f"{max_abs:.6g}"


class NanScanner:
    """Env-gated NaN/Inf scanner; see module docstring for usage."""

    def __init__(self, env_var: str = "MILES_NAN_SCAN"):
        self._env_var = env_var
        self._step = 0
        self._seen_grad_keys: set[str] = set()
        self._disabled_notice_printed = False

    def enabled(self) -> bool:
        """True when the gating env var is set to 1; use to gate compound debug blocks."""
        return os.environ.get(self._env_var) == "1"

    def step(self) -> int:
        """Bump the step= tag printed on each line; call once per microbatch, like `dumper.step()`."""
        self._step += 1
        return self._step

    def scan(self, name: str, value, *, grad: bool = False, quiet: bool = False, fatal: bool = False) -> bool:
        """Scan `value` for NaN/Inf, print stats, and return whether any was found.

        Args:
            name: Label for the printed lines.
            value: Tensor, nested list/dict of tensors (aggregated, offenders listed
                individually), nn.Module (params + main_grad/grad), or a callable
                evaluated only when scanning is enabled. None is a no-op.
            grad: Also scan the gradient when backward reaches `value` (Tensor only).
                On a boundary tensor like logits this triages a NaN's origin: value
                line fires -> forward; .grad line fires -> loss side of backward;
                neither, yet grads still blow up -> inside the model backward.
            quiet: Print only on violations.
            fatal: Raise RuntimeError on violations.
        """
        if not self.enabled():
            self._notice_disabled_once()
            return False
        if grad and not isinstance(value, Tensor):
            raise TypeError(f"scan({name!r}, grad=True) requires a Tensor, got {type(value).__name__}")
        if callable(value) and not isinstance(value, (Tensor, nn.Module)):
            value = value()
        found = self._scan(name, value, quiet)
        if found and fatal:
            raise RuntimeError(f"nan_scanner.scan({name!r}) found non-finite values, see [NAN_SCAN] lines above")
        if grad:
            self.scan_grad(name, value, quiet=quiet)
        return found

    def scan_grad(self, name: str, tensor: Tensor, *, once: bool = True, quiet: bool = True) -> None:
        """Scan the gradient of `tensor` when backward reaches it (reported as "<name>.grad").

        No-op unless `tensor` requires grad. `once` reports at most one violation
        per name to avoid per-microbatch spam.
        """
        if not self.enabled():
            self._notice_disabled_once()
            return
        if not isinstance(tensor, Tensor) or not tensor.requires_grad:
            return

        def _hook(grad: Tensor) -> None:
            key = f"{name}.grad"
            if once and key in self._seen_grad_keys:
                return
            if self._scan_tensor(key, grad, quiet) and once:
                self._seen_grad_keys.add(key)

        tensor.register_hook(_hook)

    def _notice_disabled_once(self) -> None:
        if not self._disabled_notice_printed:
            self._disabled_notice_printed = True
            print(f"[NAN_SCAN] scanning is disabled; set {self._env_var}=1 to enable", flush=True)

    def _prefix(self, name: str) -> str:
        return f"[NAN_SCAN] rank={_rank()} step={self._step} {name}"

    def _scan_tensor(self, name: str, t: Tensor, quiet: bool) -> bool:
        if t.device.type == "meta":
            print(f"{self._prefix(name)}: skipped (meta tensor)", flush=True)
            return False
        nan, inf, max_abs = _tensor_counts(t)
        found = bool(nan or inf)
        if found or not quiet:
            flag = " ***NONFINITE***" if found else ""
            print(
                f"{self._prefix(name)}: shape={tuple(t.shape)} nan={nan} inf={inf} max_abs={_fmt(max_abs)}{flag}",
                flush=True,
            )
        return found

    def _scan_leaves(self, name: str, leaves, quiet: bool) -> bool:
        """Aggregate scan over (label, tensor) pairs: one summary line, capped offender lines."""
        n = total_nan = total_inf = total_numel = 0
        max_abs: float | None = None
        offenders: list[str] = []
        for label, t in leaves:
            n += 1
            nan, inf, leaf_max = _tensor_counts(t)
            total_nan += nan
            total_inf += inf
            total_numel += t.numel()
            if leaf_max is not None:
                max_abs = leaf_max if max_abs is None else max(max_abs, leaf_max)
            if nan or inf:
                offenders.append(
                    f"{self._prefix(name)}{label}: shape={tuple(t.shape)}"
                    f" nan={nan} inf={inf} max_abs={_fmt(leaf_max)} ***NONFINITE***"
                )
        found = bool(offenders)
        if found or not quiet:
            flag = " ***NONFINITE***" if found else ""
            print(
                f"{self._prefix(name)}: tensors={n} numel={total_numel}"
                f" nan={total_nan} inf={total_inf} max_abs={_fmt(max_abs)}{flag}",
                flush=True,
            )
        for line in offenders[:_MAX_OFFENDER_LINES]:
            print(line, flush=True)
        if len(offenders) > _MAX_OFFENDER_LINES:
            print(
                f"{self._prefix(name)}: ... and {len(offenders) - _MAX_OFFENDER_LINES} more non-finite tensors",
                flush=True,
            )
        return found

    @staticmethod
    def _module_leaves(module: nn.Module):
        for param_name, param in module.named_parameters():
            yield f".{param_name}", param
            main_grad = getattr(param, "main_grad", None)
            grad = main_grad if main_grad is not None else param.grad
            if grad is not None:
                yield f".{param_name}.grad", grad

    def _scan(self, name: str, value, quiet: bool) -> bool:
        if value is None:
            return False
        if isinstance(value, Tensor):
            return self._scan_tensor(name, value, quiet)
        if isinstance(value, nn.Module):
            return self._scan_leaves(name, self._module_leaves(value), quiet)
        if isinstance(value, Mapping):
            found = False
            for key, item in value.items():
                found |= self._scan(f"{name}.{key}", item, quiet)
            return found
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) > 0 and all(isinstance(item, Tensor) for item in value):
                return self._scan_leaves(name, ((f"[{i}]", t) for i, t in enumerate(value)), quiet)
            found = False
            for i, item in enumerate(value):
                found |= self._scan(f"{name}[{i}]", item, quiet)
            return found
        return False


nan_scanner = NanScanner()
