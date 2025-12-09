from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MaximumDivisor_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3

    def update(self) -> None :
        self.N = int(self.N * 1.2 + 2)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N)]