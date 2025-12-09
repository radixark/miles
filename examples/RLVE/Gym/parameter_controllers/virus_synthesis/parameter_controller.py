from typing import Dict, List
from Gym.parameter_controller import ParameterController

class VirusSynthesis_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.loose_MAX_N = 4

    def update(self) -> None :
        self.loose_MAX_N = int(self.loose_MAX_N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(loose_MAX_N = self.loose_MAX_N)]