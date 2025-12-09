from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController


class MinimumCost_MaximumFlow_ParameterController(ParameterController):
    def __init__(self, edge_density_list: Optional[List] = None, **kwargs):
        super().__init__(**kwargs)
        self.N = 4  # Start with 4 vertices

        if edge_density_list is None :
            edge_density_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.edge_density_list = edge_density_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, edge_density = edge_density) for edge_density in self.edge_density_list]