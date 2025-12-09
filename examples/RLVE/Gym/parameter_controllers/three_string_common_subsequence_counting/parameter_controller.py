from typing import Dict, List
from Gym.parameter_controller import ParameterController

class ThreeStringCommonSubsequenceCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 3

    def update(self) -> None :
        self.MAX_N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N)]