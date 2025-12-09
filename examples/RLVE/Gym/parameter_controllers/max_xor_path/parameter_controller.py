from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class MaxXorPath_ParameterController(ParameterController) :
    def __init__(self, edge_density_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4

        if edge_density_list is None :
            edge_density_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.edge_density_list = edge_density_list

        self.MAX_bit_length = 3

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)
        while 2 ** self.MAX_bit_length < self.N * 2 :
            self.MAX_bit_length += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, edge_density = edge_density, MAX_bit_length = MAX_bit_length) for edge_density in self.edge_density_list for MAX_bit_length in range(self.MAX_bit_length, self.MAX_bit_length + 5) if int(edge_density * self.N * (self.N - 1) / 2) >= self.N]