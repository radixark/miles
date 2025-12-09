from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MatrixPermutation_BothDiagonalOne_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.OddN = 3

    def update(self) -> None :
        self.OddN += 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.OddN), dict(N = self.OddN + 1)]