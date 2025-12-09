from typing import Dict, List
from Gym.parameter_controller import ParameterController

class BinaryLinearEquation_SolutionCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_RANGE = 8
    
    def update(self) -> None :
        self.MAX_RANGE *= 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_RANGE = self.MAX_RANGE)]