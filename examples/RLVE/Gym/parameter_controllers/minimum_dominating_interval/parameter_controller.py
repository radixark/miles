from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class Minimum_DominatingInterval_ParameterController(ParameterController) :
    def __init__(self, K_density_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3

        if K_density_list is None :
            self.K_density_list = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, M = min(self.N * (self.N + 1) // 2, self.N * 2), K_density = K_density) for K_density in self.K_density_list]