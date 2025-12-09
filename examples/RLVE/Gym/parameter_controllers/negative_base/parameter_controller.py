from typing import Dict, List
from Gym.parameter_controller import ParameterController

class NegativeBase_ParameterController(ParameterController) :
    def __init__(self, MAX_R = 16, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 4
        self.MAX_R = MAX_R

    def update(self) -> None :
        self.MAX_N *= 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_R = min(self.MAX_N, self.MAX_R))]