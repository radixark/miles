from typing import Dict, List
from Gym.parameter_controller import ParameterController

class ABProgramSimulation_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4
        self.max_steps = 10

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)
        self.max_steps = int(self.max_steps * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, max_steps = self.max_steps)]