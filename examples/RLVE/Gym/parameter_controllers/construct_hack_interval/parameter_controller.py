from typing import Dict, List
from Gym.parameter_controller import ParameterController

class ConstructHackInterval_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_MOD = 10

    def update(self) -> None :
        self.MAX_MOD = int(self.MAX_MOD * 2)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_MOD = self.MAX_MOD)]