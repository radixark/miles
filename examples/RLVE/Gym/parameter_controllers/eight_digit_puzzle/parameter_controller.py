from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class EightDigitPuzzle_ParameterController(ParameterController) :
    def __init__(self, steps_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_M = 3

        self.steps_list = [2, 3, 5, 10, 15, 20, 25, 30] if steps_list is None else steps_list

    def update(self) -> None :
        self.MAX_N_M += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_M = self.MAX_N_M, steps = steps) for steps in self.steps_list]