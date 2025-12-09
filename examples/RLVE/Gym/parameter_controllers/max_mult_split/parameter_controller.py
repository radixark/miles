from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class MaxMultSplit_ParameterController(ParameterController) :
    def __init__(self, K_ratio_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3
        self.K_ratio_list = K_ratio_list if K_ratio_list is not None else [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        K_set = set([2, self.N - 1])
        for K_ratio in self.K_ratio_list :
            K = int(self.N * K_ratio)
            if 2 <= K <= self.N - 1 :
                K_set.add(K)
        return [dict(N = self.N, K = K) for K in K_set]