from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController

class Countdown_ParameterController(ParameterController) :
    def __init__(self, max_operands : Optional[List] = None, **kwargs) :
        super().__init__(**kwargs)

        self.num_operands = 3
        
        if max_operands is None :
            max_operands = [8, 16, 24, 32, 40, 48]
        self.max_operands = max_operands
    
    def update(self) -> None :
        self.num_operands += 1

    def get_parameter_list(self) -> List[Dict] :
        return [dict(num_operands = self.num_operands, max_operand = max_operand, max_target = max_operand * 10) for max_operand in self.max_operands]