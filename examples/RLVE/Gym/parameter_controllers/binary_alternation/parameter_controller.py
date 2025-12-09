from typing import Dict, List
from Gym.parameter_controller import ParameterController

class BinaryAlternation_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.zero_count = 2

    def update(self) -> None :
        self.zero_count = int(self.zero_count * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(zero_count = self.zero_count)]