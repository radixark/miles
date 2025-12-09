from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class MaximumAchromaticNumber_ParameterController(ParameterController) :
    def __init__(self, edge_density_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 5

        if edge_density_list is None :
            edge_density_list = [0.02, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        self.edge_density_list = edge_density_list

    def update(self) -> None :
        self.N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, edge_density = edge_density) for edge_density in self.edge_density_list if int(edge_density * self.N * (self.N - 1) / 2) > 0]