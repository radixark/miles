from typing import Dict, List
from Gym.parameter_controller import ParameterController

class KUR_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 8
        self.MAX_M = 8

    def update(self) -> None :
        self.MAX_N *= 2
        self.MAX_M = int(self.MAX_M * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_M = self.MAX_M)]