"""Inline NaN/Inf scanning for debugging numerical issues.

`NanScanner` prints one-line finiteness stats for a tensor, a list/dict of
tensors, or an `nn.Module` (params + grads), and reports whether any non-finite
value was found. Complementary to the Dumper (offline dump & cross-run
comparison): the scanner answers "where does the first non-finite value appear
in THIS run" — no files, no second run.

All scanning is gated on MILES_NAN_SCAN=1; when disabled every call is a no-op
(a one-time notice per process says how to enable). Call sites therefore need
no guard of their own. Use the module-level `nan_scanner` instance (mirroring
`dumper`) so state like step counters and dedup is shared across call sites:

    from miles.utils.debug_utils.nan_scan import nan_scanner

    nan_scanner.scan("logits", logits)              # print stats when enabled
    nan_scanner.scan("batch", batch, quiet=True)    # print only on violation
    nan_scanner.scan("actor", model, fatal=True)    # params + grads; raise on violation
    nan_scanner.scan_grad("dL_dlogits", logits)     # scan the gradient during backward
    nan_scanner.step()                              # bump the step= tag printed on each line

If constructing the value itself is expensive, pass a callable so it is only
evaluated when scanning is enabled:

    nan_scanner.scan("logprob_diff", lambda: [a.float() - b.float() for a, b in pairs])

Use `nan_scanner.enabled()` to gate compound debug blocks (e.g. building a
metrics dict for a custom print). When enabled, every call synchronizes the
GPU (host-side reduction).
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
        """Increment and return the step counter printed as step= on each line.

        Call once per microbatch / iteration at a natural boundary (e.g. loss
        function entry), analogous to `dumper.step()`.
        """
        self._step += 1
        return self._step

    @property
    def current_step(self) -> int:
        """The step counter printed as step= on each line."""
        return self._step

    def scan(self, name: str, value, *, quiet: bool = False, fatal: bool = False) -> bool:
        """Scan `value` for NaN/Inf and print per-tensor stats; return whether any was found.

        No-op (returns False) unless the gating env var is set.

        Args:
            name: Label printed with the stats.
            value: Tensor, nn.Module (scans params and main_grad/grad), or a nested
                Mapping/Sequence of these. A flat sequence of tensors is aggregated into
                one line, with per-element lines for offenders only. None is a no-op.
                A callable is invoked to produce the value only when scanning is enabled,
                so expensive value construction costs nothing when disabled.
            quiet: Print only when non-finite values are found.
            fatal: Raise RuntimeError if non-finite values are found.
        """
        if not self.enabled():
            self._notice_disabled_once()
            return False
        if callable(value) and not isinstance(value, (Tensor, nn.Module)):
            value = value()
        found = self._scan(name, value, quiet)
        if found and fatal:
            raise RuntimeError(f"nan_scanner.scan({name!r}) found non-finite values, see [NAN_SCAN] lines above")
        return found

    def scan_grad(self, name: str, tensor: Tensor, *, once: bool = True, quiet: bool = True) -> None:
        """Register a backward hook that scans the gradient of `tensor` when it arrives.

        No-op unless the gating env var is set.

        Args:
            name: Label for the printed stats (reported as "<name>.grad").
            tensor: Tensor to watch; no-op unless it requires grad.
            once: Report at most one violation per name to avoid per-microbatch spam.
            quiet: Print only when the gradient is non-finite.
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

    def _emit(self, name: str, shape, nan: int, inf: int, max_abs: float | None, *, quiet: bool) -> bool:
        found = bool(nan or inf)
        if found or not quiet:
            flag = " ***NONFINITE***" if found else ""
            print(
                f"{self._prefix(name)}: shape={tuple(shape)} nan={nan} inf={inf} max_abs={_fmt(max_abs)}{flag}",
                flush=True,
            )
        return found

    def _scan_tensor(self, name: str, t: Tensor, quiet: bool) -> bool:
        if t.device.type == "meta":
            print(f"{self._prefix(name)}: skipped (meta tensor)", flush=True)
            return False
        nan, inf, max_abs = _tensor_counts(t)
        return self._emit(name, t.shape, nan, inf, max_abs, quiet=quiet)

    def _scan_tensor_seq(self, name: str, seq: Sequence[Tensor], quiet: bool) -> bool:
        total_nan = total_inf = total_numel = 0
        max_abs: float | None = None
        offenders: list[str] = []
        for i, t in enumerate(seq):
            nan, inf, elem_max = _tensor_counts(t)
            total_nan += nan
            total_inf += inf
            total_numel += t.numel()
            if elem_max is not None:
                max_abs = elem_max if max_abs is None else max(max_abs, elem_max)
            if nan or inf:
                offenders.append(
                    f"{self._prefix(name)}[{i}]: shape={tuple(t.shape)}"
                    f" nan={nan} inf={inf} max_abs={_fmt(elem_max)} ***NONFINITE***"
                )
        found = bool(total_nan or total_inf)
        if found or not quiet:
            flag = " ***NONFINITE***" if found else ""
            print(
                f"{self._prefix(name)}: tensors={len(seq)} numel={total_numel}"
                f" nan={total_nan} inf={total_inf} max_abs={_fmt(max_abs)}{flag}",
                flush=True,
            )
        self._emit_offenders(name, offenders, "non-finite elements")
        return found

    def _scan_module(self, name: str, module: nn.Module, quiet: bool) -> bool:
        n_params = n_bad = 0
        offenders: list[str] = []
        for param_name, param in module.named_parameters():
            n_params += 1
            main_grad = getattr(param, "main_grad", None)
            grad = main_grad if main_grad is not None else param.grad
            for suffix, t in (("", param), (".grad", grad)):
                if t is None:
                    continue
                nan, inf, max_abs = _tensor_counts(t)
                if nan or inf:
                    n_bad += 1
                    offenders.append(
                        f"{self._prefix(name)}.{param_name}{suffix}: shape={tuple(t.shape)}"
                        f" nan={nan} inf={inf} max_abs={_fmt(max_abs)} ***NONFINITE***"
                    )
        self._emit_offenders(name, offenders, "non-finite params/grads")
        found = n_bad > 0
        if found or not quiet:
            flag = " ***NONFINITE***" if found else ""
            print(
                f"{self._prefix(name)}: module params={n_params} nonfinite_params_or_grads={n_bad}{flag}",
                flush=True,
            )
        return found

    def _emit_offenders(self, name: str, offenders: list[str], what: str) -> None:
        for line in offenders[:_MAX_OFFENDER_LINES]:
            print(line, flush=True)
        if len(offenders) > _MAX_OFFENDER_LINES:
            print(f"{self._prefix(name)}: ... and {len(offenders) - _MAX_OFFENDER_LINES} more {what}", flush=True)

    def _scan(self, name: str, value, quiet: bool) -> bool:
        if value is None:
            return False
        if isinstance(value, Tensor):
            return self._scan_tensor(name, value, quiet)
        if isinstance(value, nn.Module):
            return self._scan_module(name, value, quiet)
        if isinstance(value, Mapping):
            found = False
            for key, item in value.items():
                found |= self._scan(f"{name}.{key}", item, quiet)
            return found
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) > 0 and all(isinstance(item, Tensor) for item in value):
                return self._scan_tensor_seq(name, value, quiet)
            found = False
            for i, item in enumerate(value):
                found |= self._scan(f"{name}[{i}]", item, quiet)
            return found
        return False


nan_scanner = NanScanner()
