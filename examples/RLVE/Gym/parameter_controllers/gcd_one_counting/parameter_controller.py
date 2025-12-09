from typing import Dict, List
from Gym.parameter_controller import ParameterController

class GCDOne_Counting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_M = 5

    def update(self) -> None :
        self.MAX_N_M = int(self.MAX_N_M * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_M = self.MAX_N_M)]