from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class SAT_ParameterController(ParameterController) :
    def __init__(self, density_list : Optional[List] = None, M_multiple_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3

        if density_list is None :
            density_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        self.density_list = density_list

        if M_multiple_list is None :
            M_multiple_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
        self.M_multiple_list = M_multiple_list

    def update(self) -> None :
        self.N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, M = int(M_multiple * self.N), density = density) for density in self.density_list for M_multiple in self.M_multiple_list]