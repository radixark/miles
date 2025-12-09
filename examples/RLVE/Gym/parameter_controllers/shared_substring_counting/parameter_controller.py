from typing import Dict, List
from Gym.parameter_controller import ParameterController

class SharedSubstringCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_LEN = 10

    def update(self) -> None :
        self.MAX_LEN = int(self.MAX_LEN * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_LEN = self.MAX_LEN)]