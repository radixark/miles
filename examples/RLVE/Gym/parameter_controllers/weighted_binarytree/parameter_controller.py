from typing import Dict, List
from Gym.parameter_controller import ParameterController

class WeightedBinaryTree_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.MAX_SCORE = 5

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, MAX_SCORE = self.MAX_SCORE)]