from typing import Dict, List
from Gym.parameter_controller import ParameterController

class CirculatingGrid_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_R_C = 3

    def update(self) -> None :
        self.MAX_R_C += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_R_C = self.MAX_R_C)]