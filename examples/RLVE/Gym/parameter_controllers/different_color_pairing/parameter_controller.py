from typing import Dict, List
from Gym.parameter_controller import ParameterController

class DifferentColorPairing_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 6

    def update(self) -> None :
        self.N += 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N)]