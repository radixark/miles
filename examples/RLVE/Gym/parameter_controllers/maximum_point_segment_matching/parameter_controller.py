from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MaximumPointSegmentMatching_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_C_N = 3

    def update(self) -> None :
        self.MAX_C_N = int(self.MAX_C_N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_C_N = self.MAX_C_N)]