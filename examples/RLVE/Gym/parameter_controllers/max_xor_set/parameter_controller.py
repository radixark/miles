from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MaxXorSet_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.MAX_bit_length = 3

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)
        while (2 ** self.MAX_bit_length - 2) < self.N * 2 :
            self.MAX_bit_length += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, MAX_bit_length = MAX_bit_length) for MAX_bit_length in range(self.MAX_bit_length, self.MAX_bit_length + 5)]