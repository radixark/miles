from typing import Dict, List
from Gym.parameter_controller import ParameterController

class SumSetMultiplication_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 3
        self.MAX_K = 8

    def update(self) -> None :
        self.MAX_N += 1
        self.MAX_K = int(self.MAX_K * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_K = self.MAX_K)]