from typing import Dict, Iterable, Callable, Tuple

import torch

NamedParametersGetter = Callable[[], Iterable[Tuple[str, torch.Tensor]]]


class TensorBackuper:
    def __init__(self, source_getter: NamedParametersGetter):
        self._source_getter = source_getter

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        for name, param in self._source_getter():
            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        for name, param in self._source_getter():
            assert name in params_dict
            param.copy_(params_dict[name], non_blocking=True)
        torch.cuda.synchronize()
