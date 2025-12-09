from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class GaussianElimination_ParameterController(ParameterController) :
    def __init__(self, M_multiple_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        if M_multiple_list is None :
            M_multiple_list = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
        self.M_multiple_list = M_multiple_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, M = int(M_multiple * self.N)) for M_multiple in self.M_multiple_list]