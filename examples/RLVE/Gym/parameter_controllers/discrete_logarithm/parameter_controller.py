from typing import Dict, List
from Gym.parameter_controller import ParameterController

class DiscreteLogarithm_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_Z = 10

    def update(self) -> None :
        self.MAX_Z = int(self.MAX_Z * 1.2 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_Z = self.MAX_Z)]