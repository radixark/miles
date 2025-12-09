from typing import Dict, List
from Gym.parameter_controller import ParameterController

class BannedPointSupersetPathCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_M_R = 1
        self.MAX_O = 10

    def update(self) -> None :
        self.MAX_N_M_R = int(self.MAX_N_M_R * 1.1 + 1)
        self.MAX_O = int(self.MAX_O * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_M_R = self.MAX_N_M_R, MAX_O = self.MAX_O)]