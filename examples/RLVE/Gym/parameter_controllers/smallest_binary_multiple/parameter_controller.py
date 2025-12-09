from typing import Dict, List
from Gym.parameter_controller import ParameterController

class SmallestBinaryMultiple_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_A = 10

    def update(self) -> None :
        self.MAX_A = int(self.MAX_A * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_A = self.MAX_A)]