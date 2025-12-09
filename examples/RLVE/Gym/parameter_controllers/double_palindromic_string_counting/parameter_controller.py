from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class DoublePalindromicStringCounting_ParameterController(ParameterController) :
    def __init__(self, C_List : Optional[List[int]] = None, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N = 3
        self.C_List = C_List if C_List is not None else [2, 3, 4, 5]

    def update(self) -> None :
        self.MAX_N = int(self.MAX_N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N = self.MAX_N, C = C) for C in self.C_List]