from typing import Dict, List
from Gym.parameter_controller import ParameterController

class CRT_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_X = 5
        self.MAX_M = 2

        self.current_stage = 0

    def update(self) -> None :
        self.current_stage += 1
        if self.current_stage % 3 == 0 :
            self.MAX_M += 1
        else :
            self.MAX_X = int(self.MAX_X * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_X = self.MAX_X, M = M) for M in range(2, self.MAX_M + 1)]