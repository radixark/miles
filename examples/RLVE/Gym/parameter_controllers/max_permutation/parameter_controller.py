from typing import Dict, List
from Gym.parameter_controller import ParameterController

class MaxPermutation_ParameterController(ParameterController) :
    def __init__(self, MAX_DIGIT_NUM : int = 5, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.MAX_DIGIT_NUM = MAX_DIGIT_NUM

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, MAX_DIGIT_NUM = self.MAX_DIGIT_NUM)]