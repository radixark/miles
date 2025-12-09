from typing import Dict, List
from Gym.parameter_controller import ParameterController

class NextPalindromic_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.digit_num = 2

    def update(self) -> None :
        self.digit_num = int(self.digit_num * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(digit_num = self.digit_num)]