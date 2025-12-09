from typing import Dict, List
from Gym.parameter_controller import ParameterController

class KingSorting_ParameterController(ParameterController) :
    def __init__(self, MAX_A_B : int = 10, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.MAX_A_B = MAX_A_B

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, MAX_A_B = self.MAX_A_B)]