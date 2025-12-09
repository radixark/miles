from typing import Dict, List
from Gym.parameter_controller import ParameterController

class ChoHamsters_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 1
        self.MAX_M = 4

    def update(self) -> None :
        self.N += 1
        self.MAX_M = int(self.MAX_M * 1.5 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, MAX_M = self.MAX_M)]