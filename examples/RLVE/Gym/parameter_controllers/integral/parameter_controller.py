from typing import Dict, List
from Gym.parameter_controller import ParameterController


class Integral_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.node_num = 2

    def update(self) -> None :
        self.node_num += 1

    def get_parameter_list(self) -> List[Dict] :
        return [{"node_num" : self.node_num}]
