from typing import Dict, List
from Gym.parameter_controller import ParameterController

class AddMultiple_Divisible_Counting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 16

    def update(self) -> None :
        self.MAX_N = int(self.MAX_N * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N)]