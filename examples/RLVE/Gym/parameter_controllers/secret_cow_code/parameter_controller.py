from typing import Dict, List
from Gym.parameter_controller import ParameterController

class SecretCowCode_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_K = 8
        self.MAX_N = 5

    def update(self) -> None :
        self.MAX_N = int(self.MAX_N * 1.1 + 1)
        self.MAX_K *= 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, MAX_K = self.MAX_K)]