from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class SpyNetwork_ParameterController(ParameterController) :
    def __init__(self, edge_density_list : Optional[List] = None, dominated_probability_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4

        if edge_density_list is None :
            edge_density_list = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15]
        self.edge_density_list = edge_density_list

        if dominated_probability_list is None :
            dominated_probability_list = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.dominated_probability_list = dominated_probability_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, edge_density = edge_density, dominated_probability = dominated_probability) for edge_density in self.edge_density_list for dominated_probability in self.dominated_probability_list if int(edge_density * self.N * (self.N - 1)) > 0]