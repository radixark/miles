---
paths:
  - "miles/**/*.py"
  - "train.py"
  - "train_async.py"
---

# Multi-Hardware

Miles is a multi-hardware training framework: the same training path runs on
CUDA GPUs and on TPU (torch_tpu), and more accelerators may follow. The platform
is chosen once per process by `MILES_ACCELERATOR` and exposed as
`miles.platforms.current_platform`. Apply this rule to new and modified code.

## Do not hard-code CUDA

- **No literal `"cuda"` device in shared code.** Use
  `current_platform.current_device()` for the device and
  `current_platform.device_type` for the type string, never `"cuda"`,
  `"cuda:0"`, `torch.device("cuda")`, `.cuda()` or `device="cuda"`.
- **No hard-coded CUDA collectives.** Do not write `backend="nccl"` or a
  `"cuda:<backend>"` device-backend string in shared code; take the backend
  from the run's arguments or the platform.
- **Use `torch.accelerator` for what it covers.** Synchronize, empty the cache,
  read memory stats and the device index with `torch.accelerator.*`, not
  `torch.cuda.*`.
- **Put what differs in the platform, not at the call site.** When a call has no
  `torch.accelerator` equivalent (device binding, device name, compile support),
  add a method or capability attribute on `Platform` with the CUDA behavior, and
  override it in the other platform. Do not add new `if IS_TPU:` /
  `if torch.cuda.is_available():` branches to shared code; `IS_TPU` is legacy
  that is being removed.

## When CUDA-only code is fine

CUDA-specific calls are allowed where the code can only run on CUDA, and that
condition is explicit:

- inside the CUDA implementation of `Platform` in `miles/platforms/`;
- in a module that only CUDA runs reach, such as the Megatron backend and the
  Triton / CUDA kernels under `miles/kernels/`, as long as shared code imports
  it only on a CUDA path;
- behind a feature or argument that requires CUDA and is validated at startup,
  so other platforms fail early with a clear error instead of at the call.

Even there, prefer the device-agnostic API when one exists. A new CUDA-only
module states so in its docstring, and it must not be imported at module
level by code the other platforms run, since `torch_tpu` processes may lack
CUDA libraries.
