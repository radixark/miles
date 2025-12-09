from typing import Dict, List
from Gym.parameter_controller import ParameterController

class EuclidGame_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_X_Y = 16

    def update(self) -> None :
        self.MAX_X_Y *= 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_X_Y = self.MAX_X_Y)]