from typing import Dict, List
from Gym.parameter_controller import ParameterController

class PanSolarPanels_ParameterController(ParameterController) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.MAX_A_B_C_D = 10

    def update(self) -> None :
        self.MAX_A_B_C_D = int(self.MAX_A_B_C_D * 1.5)

    def get_parameter_list(self) -> List[Dict] :
        return [dict(MAX_A_B_C_D = self.MAX_A_B_C_D)]