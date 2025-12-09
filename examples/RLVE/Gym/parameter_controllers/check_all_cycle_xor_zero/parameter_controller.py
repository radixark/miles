from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class CheckAllCycleXorZero_ParameterController(ParameterController) :
    def __init__(self, edge_ratio_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4

        if edge_ratio_list is None :
            edge_ratio_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
        self.edge_ratio_list = edge_ratio_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, edge_ratio = edge_ratio) for edge_ratio in self.edge_ratio_list if int(self.N * edge_ratio) <= self.N * (self.N - 1) // 2]