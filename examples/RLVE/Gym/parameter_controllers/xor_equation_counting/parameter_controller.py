from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class XorEquationCounting_ParameterController(ParameterController) :
    def __init__(self, RANGE_List : Optional[List[int]] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 2
        self.RANGE_List = RANGE_List if RANGE_List is not None else [2 ** 2 - 1, 2 ** 3 - 1, 2 ** 5 - 1, 2 ** 7 - 1, 2 ** 10 - 1, 2 ** 12 - 1, 2 ** 15 - 1, 2 ** 17 - 1, 2 ** 20 - 1]

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, RANGE = RANGE) for RANGE in self.RANGE_List]