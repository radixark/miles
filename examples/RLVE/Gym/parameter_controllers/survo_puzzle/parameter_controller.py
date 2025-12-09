from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class SurvoPuzzle_ParameterController(ParameterController) :
    def __init__(self, sparsity_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_M = 3

        if sparsity_list is None :
            sparsity_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.sparsity_list = sparsity_list

    def update(self) -> None :
        self.MAX_N_M += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_M = self.MAX_N_M, sparsity = sparsity) for sparsity in self.sparsity_list]