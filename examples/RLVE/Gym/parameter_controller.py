from typing import Dict, List
from abc import ABC, abstractmethod



class ParameterController(ABC) :
    """
    Abstract base for driving the sequence of `parameter` dicts fed into a VerifiableEnvironment.generator(seed, parameter) call.
    """

    def __init__(self) :
        pass


    @abstractmethod
    def update(self) -> None :
        """
        Advance to the next parameter setting and store it
        """
        pass


    @abstractmethod
    def get_parameter_list(self) -> List[Dict] :
        """
        Returns the full list of parameter dicts this controller manages.
        """
        pass
