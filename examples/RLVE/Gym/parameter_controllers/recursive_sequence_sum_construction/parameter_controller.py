from typing import Dict, List
from Gym.parameter_controller import ParameterController

class RecursiveSequenceSumConstruction_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.N = 4
        
        self.MAX_F0 = 128
        self.MAX_A = 16
        self.MAX_B = 16384

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, MAX_F0 = self.MAX_F0, MAX_A = self.MAX_A, MAX_B = self.MAX_B)]