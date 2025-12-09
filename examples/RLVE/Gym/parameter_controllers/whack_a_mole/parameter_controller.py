from typing import Dict, List
from Gym.parameter_controller import ParameterController

class WhackAMole_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_M = 3

    def update(self) -> None :
        self.MAX_N_M = int(self.MAX_N_M * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_M = self.MAX_N_M)]