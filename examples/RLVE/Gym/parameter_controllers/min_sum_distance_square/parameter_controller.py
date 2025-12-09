from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MinSumDistanceSquare_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.M = 2

    def update(self) -> None :
        self.M = int(self.M * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(M = self.M)]