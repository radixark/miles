from typing import Dict, List
from Gym.parameter_controller import ParameterController

class Path_NoGoingBack_Counting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_M = 10

    def update(self) -> None :
        self.MAX_M = int(self.MAX_M * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_M = self.MAX_M)]