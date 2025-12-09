from typing import Dict, List
from Gym.parameter_controller import ParameterController

class StoneGame_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_SUM = 5

    def update(self) -> None :
        self.MAX_SUM = int(self.MAX_SUM * 1.2 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_SUM = self.MAX_SUM)]