from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MinKDivisorNumber_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_K = 5

    def update(self) -> None :
        self.MAX_K *= 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_K = self.MAX_K)]