from collections import defaultdict
from typing import Dict, Iterable, Callable, Tuple, Any

import torch

NamedParametersGetter = Callable[[], Iterable[Tuple[str, torch.Tensor]]]


class TensorBackuper:
    def __init__(self, source_getter: NamedParametersGetter):
        self._source_getter = source_getter
        self._backups: Dict[str, Dict[str, torch.Tensor]] = defaultdict()

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()
