import random
from typing import Dict, List
from Gym.parameter_controller import ParameterController


class StringReversalConstruction_ParameterController(ParameterController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 3

    def update(self) -> None:
        self.n += 1

    def get_parameter_list(self) -> List[Dict]:
        return [dict(n=self.n)] 