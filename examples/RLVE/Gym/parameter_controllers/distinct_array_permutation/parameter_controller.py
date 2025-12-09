from typing import Dict, List
from Gym.parameter_controller import ParameterController

class DistinctArrayPermutation_ParameterController(ParameterController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.N = 3

    def update(self) -> None:
        self.N += 1
        self.N = min(self.N, 22)

    def get_parameter_list(self) -> List[Dict]:
        return [dict(N = self.N)]