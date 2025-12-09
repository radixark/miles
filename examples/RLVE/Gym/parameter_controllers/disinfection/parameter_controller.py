from typing import Dict, List
from Gym.parameter_controller import ParameterController

class Disinfection_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_A_B_C = 2

    def update(self) -> None :
        self.MAX_A_B_C += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_A_B_C = self.MAX_A_B_C)]