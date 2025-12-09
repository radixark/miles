from typing import Dict, List
from Gym.parameter_controller import ParameterController

class PowerNest_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.max_number = 4

    def update(self) -> None :
        self.max_number *= 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(max_number = self.max_number,)]