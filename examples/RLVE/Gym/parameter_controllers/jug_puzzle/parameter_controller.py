from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class JugPuzzle_ParameterController(ParameterController) :
    def __init__(self, steps_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3

        self.steps_list = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50] if steps_list is None else steps_list

    def update(self) -> None :
        self.N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, steps = steps) for steps in self.steps_list]