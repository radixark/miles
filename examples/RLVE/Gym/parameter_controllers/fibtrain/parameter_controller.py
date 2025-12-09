from typing import Dict, List
from Gym.parameter_controller import ParameterController

class Fibtrain_ParameterController(ParameterController) :
    def __init__(self, MAX_A_B = 20, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 5
        self.MAX_A_B = MAX_A_B

    def update(self) -> None :
        self.MAX_N = int(self.MAX_N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_A_B = self.MAX_A_B)]