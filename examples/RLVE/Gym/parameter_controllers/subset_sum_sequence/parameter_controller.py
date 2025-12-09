from typing import Dict, List
from Gym.parameter_controller import ParameterController

class SubsetSumSequence_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 3
        self.MAX_K = 2

    def update(self) -> None :
        self.MAX_N *= 2
        self.MAX_K += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_K = min(self.MAX_N, self.MAX_K))]