from typing import Dict, List
from Gym.parameter_controller import ParameterController


class RepeatSequenceLNDS_ParameterController(ParameterController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 3
        self.MAX_T = 5
        
    def update(self) -> None:
        self.n += 1
        self.MAX_T = int(self.MAX_T * 1.5)

    def get_parameter_list(self) -> List[Dict]:
        return [dict(n=self.n, MAX_T=self.MAX_T)] 