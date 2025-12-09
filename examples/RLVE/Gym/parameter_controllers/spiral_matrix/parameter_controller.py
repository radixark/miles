from Gym.parameter_controller import ParameterController
from typing import Dict, List


class SpiralMatrix_ParameterController(ParameterController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MAX_M_N = 3

    def update(self) -> None:
        self.MAX_M_N = int(self.MAX_M_N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict]:
        return [dict(MAX_M_N = self.MAX_M_N)]
