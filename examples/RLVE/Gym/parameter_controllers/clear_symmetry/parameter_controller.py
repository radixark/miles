from typing import Dict, List
from Gym.parameter_controller import ParameterController


class ClearSymmetry_ParameterController(ParameterController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MAX_X = 5

    def update(self) -> None:
        self.MAX_X = int(self.MAX_X * 1.5)

    def get_parameter_list(self) -> List[Dict]:
        return [dict(MAX_X=self.MAX_X)]