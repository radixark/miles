from typing import Dict, List
from Gym.parameter_controller import ParameterController

class GcdLcmCounting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_LCM = 10

    def update(self) -> None :
        self.MAX_LCM = int(self.MAX_LCM * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_LCM = self.MAX_LCM)]