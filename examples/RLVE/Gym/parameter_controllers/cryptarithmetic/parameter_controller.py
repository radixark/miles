from typing import Dict, List
from Gym.parameter_controller import ParameterController

class Cryptarithmetic_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3

    def update(self) -> None :
        self.N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, addend_length = addend_length) for addend_length in range(self.N + 3, max(self.N + 3, self.N * 2) + 1)]