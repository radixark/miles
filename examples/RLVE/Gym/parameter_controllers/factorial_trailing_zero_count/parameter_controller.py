from typing import Dict, List
from Gym.parameter_controller import ParameterController

class FactorialTrailingZeroCount_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_K = 10

    def update(self) -> None :
        self.MAX_N_K = int(self.MAX_N_K * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_K = self.MAX_N_K)]