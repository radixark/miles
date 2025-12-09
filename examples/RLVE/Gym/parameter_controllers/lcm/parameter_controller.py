from typing import Dict, List
from Gym.parameter_controller import ParameterController

class LCM_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_a_b = 15

    def update(self) -> None :
        self.MAX_a_b = int(self.MAX_a_b * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_a_b = self.MAX_a_b)]