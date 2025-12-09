from typing import Dict, List
from Gym.parameter_controller import ParameterController

class BitAndZero_PathCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.max_length = 3

    def update(self) -> None :
        self.max_length = int(self.max_length * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(max_length = self.max_length)]