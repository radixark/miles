from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class Maze_ParameterController(ParameterController) :
    def __init__(self, density_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4

        if density_list is None :
            density_list = [0.1, 0.2, 0.3, 0.4]
        self.density_list = density_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, density = density) for density in self.density_list]