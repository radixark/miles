from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MinXorPair_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.max_bit_length = 3

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)
        while (1 << self.max_bit_length) <= self.N * 2 :
            self.max_bit_length += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, max_bit_length = self.max_bit_length)]