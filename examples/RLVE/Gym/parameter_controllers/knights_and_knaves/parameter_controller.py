from typing import Dict, List
from Gym.parameter_controller import ParameterController

class KnightsAndKnaves_ParameterController(ParameterController):
    def __init__(self, depth_constraint: int = 2, width_constraint: int = 2, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.depth_constraint = depth_constraint
        self.width_constraint = width_constraint

    def update(self) -> None :
        self.N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, depth_constraint=self.depth_constraint, width_constraint=self.width_constraint)]