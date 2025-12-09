from typing import Dict, List
from Gym.parameter_controller import ParameterController

class BoundedSubarrayCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.M = 2

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)
        self.M = int(self.M * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, M = self.M)]