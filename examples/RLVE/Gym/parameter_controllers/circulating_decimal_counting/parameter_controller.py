from typing import Dict, List
from Gym.parameter_controller import ParameterController

class CirculatingDecimalCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 5
        self.MAX_M = 5
        self.MAX_K = 3

    def update(self) -> None :
        self.MAX_N = int(self.MAX_N * 1.5)
        self.MAX_M = int(self.MAX_M * 1.5)
        self.MAX_K = int(self.MAX_K * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_M = self.MAX_M, MAX_K = self.MAX_K)]