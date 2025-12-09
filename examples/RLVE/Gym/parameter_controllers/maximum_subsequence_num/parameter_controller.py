from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class Maximum_SubsequenceNum_ParameterController(ParameterController) :
    def __init__(self, K_ratio_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 2
        self.M = 2

        if K_ratio_list is None :
            K_ratio_list = [0.1, 0.2, 0.3, 0.5, 1.0]
        self.K_ratio_list = K_ratio_list

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)
        self.M = int(self.M * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        Ks = set()
        Ks.add(2)
        for K_ratio in self.K_ratio_list :
            K = int(self.N * K_ratio)
            if K >= 2 :
                Ks.add(K)
        return [dict(N = self.N, M = self.M, K = K) for K in Ks]