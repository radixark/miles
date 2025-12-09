from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class KPartition_ParameterController(ParameterController) :
    def __init__(self, Ks : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.M = 2  # N//K
        if Ks is None :
            self.Ks = [2, 3, 4, 5]
        else :
            self.Ks = Ks

    def update(self) -> None :
        self.M += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(N = self.M * K, K = K) for K in self.Ks]