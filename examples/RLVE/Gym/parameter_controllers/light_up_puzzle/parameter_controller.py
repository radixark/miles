from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class LightUpPuzzle_ParameterController(ParameterController) :
    def __init__(self, density_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_N_M = 3

        if density_list is None :
            density_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.density_list = density_list

    def update(self) -> None :
        self.MAX_N_M += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_N_M = self.MAX_N_M, density = density) for density in self.density_list]