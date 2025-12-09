from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class Bridge_ParameterController(ParameterController) :
    def __init__(self, edge_density_list : Optional[List] = None, component_num_density_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4

        if edge_density_list is None :
            edge_density_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.edge_density_list = edge_density_list

        if component_num_density_list is None :
            component_num_density_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
        self.component_num_density_list = component_num_density_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        component_nums = set()
        for component_num_density in self.component_num_density_list :
            component_num = int(component_num_density * self.N)
            if component_num >= 2 :
                component_nums.add(component_num)
        return [dict(N = self.N, edge_density = edge_density, component_num = component_num) for edge_density in self.edge_density_list for component_num in component_nums]