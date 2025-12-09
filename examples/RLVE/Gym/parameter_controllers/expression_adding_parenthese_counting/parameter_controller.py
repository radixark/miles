from typing import Dict, List
from Gym.parameter_controller import ParameterController

class Expression_AddingParenthese_Counting_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.num_operands = 3
    
    def update(self) -> None :
        self.num_operands += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(num_operands = self.num_operands)]