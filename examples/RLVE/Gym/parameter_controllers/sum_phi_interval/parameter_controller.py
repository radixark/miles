from typing import Dict, List
from Gym.parameter_controller import ParameterController

class SumPHIInterval_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_DELTA = 5

    def update(self) -> None :
        self.MAX_DELTA = int(self.MAX_DELTA * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_DELTA = self.MAX_DELTA)]