from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class PowerShortcut_ParameterController(ParameterController) :
    def __init__(self, edge_density_list : Optional[List] = None, K_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4

        if edge_density_list is None :
            edge_density_list = [0.01, 0.02, 0.03]
        self.edge_density_list = edge_density_list

        if K_list is None :
            K_list = [1, 2, 3]
        self.K_list = K_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, edge_density = edge_density, K = K) for edge_density in self.edge_density_list for K in self.K_list]