from typing import Dict, List
from Gym.parameter_controller import ParameterController

class FixedModK_Selection_Counting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 8
        self.MAX_K = 5

    def update(self) -> None :
        self.MAX_N *= 2
        self.MAX_K = int(self.MAX_K * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_K = self.MAX_K)]