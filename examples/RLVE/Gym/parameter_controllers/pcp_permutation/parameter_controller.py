from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class PCPPermutation_ParameterController(ParameterController) :
    def __init__(self, average_length_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3

        if average_length_list is None :
            average_length_list = [1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.0, 10.0]
        self.average_length_list = average_length_list

    def update(self) -> None :
        self.N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, average_length = average_length) for average_length in self.average_length_list]