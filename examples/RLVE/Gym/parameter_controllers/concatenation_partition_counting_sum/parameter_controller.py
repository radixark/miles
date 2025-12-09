from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class ConcatenationPartitionCountingSum_ParameterController(ParameterController) :
    def __init__(self, M_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 2
        self.M_list = M_list if M_list is not None else [1, 2, 3, 4, 5]

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, M = M) for M in self.M_list]