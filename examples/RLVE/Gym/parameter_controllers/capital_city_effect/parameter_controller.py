from typing import Dict, List
from Gym.parameter_controller import ParameterController

class CapitalCityEffect_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_R = 20

    def update(self) -> None :
        self.MAX_R = int(self.MAX_R * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_R = self.MAX_R)]