from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class MinimumIntervalCoverage_ParameterController(ParameterController) :
    def __init__(self, M_multiple_list : Optional[List[float]] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.M_multiple = M_multiple_list if M_multiple_list is not None else [0.7, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, M = int(M_multiple * self.N)) for M_multiple in self.M_multiple]