from typing import Dict, List
from Gym.parameter_controller import ParameterController

class LampChanging_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_T = 8

    def update(self) -> None :
        self.MAX_N_T = int(self.MAX_N_T * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_T = self.MAX_N_T)]