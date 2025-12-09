from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class AndOr_Sequence_Counting_ParameterController(ParameterController) :
    def __init__(self, M_list : Optional[List[int]] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 2
        self.M_List = M_list if M_list is not None else list(range(1, 20 + 1))

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, M = M) for M in self.M_List]