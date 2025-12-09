from typing import Dict, List
from Gym.parameter_controller import ParameterController

class PairMoreOneCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_M = 10
        self.MAX_delta = 5

    def update(self) -> None :
        self.MAX_M *= 2
        self.MAX_delta = int(self.MAX_delta * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_M = self.MAX_M, MAX_delta = self.MAX_delta)]