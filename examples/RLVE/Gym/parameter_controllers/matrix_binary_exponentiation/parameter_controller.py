from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class Matrix_BinaryExponentiation_ParameterController(ParameterController) :
    def __init__(self, N_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_K = 2
        self.N_list = N_list if N_list is not None else [2, 3, 5, 10]

    def update(self) -> None :
        self.MAX_K *= 2

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = N, MAX_K = self.MAX_K) for N in self.N_list]