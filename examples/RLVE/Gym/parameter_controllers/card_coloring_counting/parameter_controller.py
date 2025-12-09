from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController
import math

class CardColoringCounting_ParameterController(ParameterController) :
    def __init__(self, Ks : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        if Ks == None :
            self.Ks = [0, 1, 2, 3]
        else :
            self.Ks = Ks

    def update(self) -> None :
        self.N += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.N, K = K) for K in self.Ks if K < math.factorial(self.N)]