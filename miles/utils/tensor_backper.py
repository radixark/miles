from typing import Dict

import torch


class TensorBackuper:
    @torch.no_grad()
    def backup(self, params_dict: Dict[str, torch.Tensor]) -> None:
        for name, param in named_parameters(self.args, self.model):
            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def restore(self, params_dict: Dict[str, torch.Tensor]) -> None:
        for name, param in named_parameters(self.args, self.model):
            assert name in params_dict
            param.copy_(params_dict[name], non_blocking=True)
        torch.cuda.synchronize()
