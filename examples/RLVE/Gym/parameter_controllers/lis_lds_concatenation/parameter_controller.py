from typing import Dict, List
from Gym.parameter_controller import ParameterController

class LIS_LDS_Concatenation_ParameterController(ParameterController) :
    def __init__(self, MAX : int = 100000, **kwargs) :
        super().__init__(**kwargs)
        self.N = 5
        self.MAX = MAX

    def update(self) -> None :
        self.N = int(self.N * 1.2)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, MAX = self.MAX)]