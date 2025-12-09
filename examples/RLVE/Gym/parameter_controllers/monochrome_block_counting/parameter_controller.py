from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MonochromeBlockCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_A_B = 5

    def update(self) -> None :
        self.MAX_A_B = int(self.MAX_A_B * 1.5 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_A_B = self.MAX_A_B)]