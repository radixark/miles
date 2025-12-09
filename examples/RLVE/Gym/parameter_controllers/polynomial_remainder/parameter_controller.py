from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class PolynomialRemainder_ParameterController(ParameterController) :
    def __init__(self, M_list : Optional[List] = None, Mratio_list : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)
        self.N = 3

        self.M_list = M_list if M_list is not None else [2, 3, 4, 5]
        self.Mratio_list = Mratio_list if Mratio_list is not None else [0.1, 0.3, 0.5, 0.7, 0.9]

    def update(self) -> None :
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict] :
        Ms = set()
        for M in self.M_list :
            if M <= self.N :
                Ms.add(M)
        for Mratio in self.Mratio_list :
            M = int(self.N * Mratio)
            if M >= 2 :
                Ms.add(M)
        return [dict(N = self.N, M = M) for M in Ms]