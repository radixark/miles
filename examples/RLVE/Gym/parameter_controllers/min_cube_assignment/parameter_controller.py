from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MinCubeAssignment_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_P_Q_R = 2

    def update(self) -> None :
        self.MAX_P_Q_R += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_P_Q_R = self.MAX_P_Q_R)]