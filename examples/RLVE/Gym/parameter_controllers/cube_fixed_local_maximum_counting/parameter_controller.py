from typing import Dict, List
from Gym.parameter_controller import ParameterController

class Cube_FixedLocalMaximumCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_M_L = 3

    def update(self) -> None :
        self.MAX_N_M_L = int(self.MAX_N_M_L * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_M_L = self.MAX_N_M_L)]