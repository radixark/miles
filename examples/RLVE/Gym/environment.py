import os
import re
import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any, Union



import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

class TimeoutException(Exception) :
    pass

def timeout(seconds) :
    def decorator(func) :
        @functools.wraps(func)
        def wrapper(*args, **kwargs) :
            executor = ThreadPoolExecutor(max_workers = 1)
            future = executor.submit(func, *args, **kwargs)
            try :
                return future.result(timeout=seconds)
            except FutureTimeoutError :
                raise TimeoutException("Function timed out after {} seconds".format(seconds))
            finally :
                executor.shutdown(wait=False, cancel_futures=True)
        return wrapper
    return decorator



import torch
import random
import numpy as np
def manual_seed(args_or_seed : int, fix_cudnn = False) :
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if fix_cudnn :
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa



class VerifiableEnvironment(ABC) :
    """
    Abstract base class for a verifiable environment.
    """
    def __init__(self, answer_markers : Optional[Tuple[str, str]] = None) :
        """
        Initializes the environment with default seed and parameter values.
        """
        self.seed = None
        self.parameter = None

        if answer_markers is None :
            answer_markers = (r"<answer>", r"</answer>")
        assert len(answer_markers) == 2 and isinstance(answer_markers[0], str) and isinstance(answer_markers[1], str), "answer_markers should be a tuple of two strings"
        self.answer_markers = answer_markers

        self.passing_reward_threshold = 1.0


    def generator(self, seed : int, parameter : Optional[Dict] = None, timeout_second : int = 10) -> bool :
        """
        Initializes the environment with the given seed and (initial) parameters, and samples environment-specific parameters to generate a problem.

        Args:
            seed (int): Random seed for reproducibility.
            parameter (Optional[Dict]): Dictionary of (initial) problem parameters.
            timeout_second (int): Timeout in seconds for the generation process.

        Returns:
            bool: True if the generation was successful, False otherwise.
        """
        @timeout(timeout_second)
        def self_generate() :
            self.seed = seed
            self.parameter = copy.deepcopy(parameter) if parameter is not None else {}

            manual_seed(self.seed)
            self._generate()
        try :
            self_generate()
        except :
            return False
        return self.parameter is not None


    @abstractmethod
    def _generate(self) -> None :
        """
        Subclasses must implement problem generation using self.seed and self.parameter.
        """
        pass


    def prompt_generator(self) -> str :
        """
        Generates the prompt string for the problem.

        Returns:
            str: The formatted prompt for the problem.
        """
        assert self.seed is not None and self.parameter is not None, "generator() should be called before prompt_generator()"

        return self._prompt_generate()


    @abstractmethod
    def _prompt_generate(self) -> str :
        """
        Subclasses must implement prompt generation using self.seed and self.parameter.

        Returns:
            str: The problem prompt.
        """
        pass


    def processor(self, output : str) -> Any :
        """
        Processes the model's output to extract useful information.

        Args:
            output (str): The string output from a model.

        Returns:
            Any: Any useful information that may be used for following steps (e.g., scoring).
        """
        
        # Remove everything before the first "Assistant:" (if possible)
        if "Assistant:" in output :
            output = output.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in output :
            output = output.split("<|im_start|>assistant", 1)[1]
        else :
            pass

        answer_pattern = re.escape(self.answer_markers[0]) + r"(.*?)" + re.escape(self.answer_markers[1])
        matches = list(re.finditer(answer_pattern, output, re.DOTALL))
        if matches :
            answer = matches[-1].group(1)
        else :
            answer = None
        return self._process(answer)


    @abstractmethod
    def _process(self, answer : Optional[str]) -> Any :
        """
        Subclasses must implement the processing of the answer.

        Args:
            answer (str): The model's answer. If it is None, it means the model did not provide an answer in the expected format.

        Returns:
            Any: The processed answer, which may be used for scoring.
        """
        pass


    @abstractmethod
    def scorer(self, output : str) -> float :
        """
        Computes a numeric score for the output, which should be in [-1.0, +1.0].

        Args:
            output (str): The model's output.

        Returns:
            float: The score for the given output, between -1.0 and +1.0.
        """
        pass


    def verifier(self, output : str) -> Dict[str, Union[float, int]] :
        """
        Verifies the model's output.
        """
        try :
            score = self.scorer(output)
        except :
            score = -1.0
        assert -1.0 <= score <= +1.0, "Score out of bounds: score={}\n\nPrompt:\n{}".format(score, self.prompt_generator())
        
        eps = 1E-6
        return dict(
            reward = score, # [-1.0, +1.0]
            accuracy = int(score >= self.passing_reward_threshold - eps), # 0 or 1
            format_score = int(score >= -1.0 + eps), # 0 or 1
        )


    def get_config(self) -> Dict :
        """
        Returns the configuration of the current problem.

        Returns:
            Dict: Dictionary with keys 'seed' and 'parameter'.
        """
        return dict(seed = self.seed, parameter = self.parameter, passing_reward_threshold = self.passing_reward_threshold)


    def set_config(self, config : Dict) -> None :
        """
        Sets the configuration for the current problem.

        Args:
            config (Dict): Dictionary with 'seed' and 'parameter' keys.
        """
        assert "seed" in config, "seed is required in config"
        assert "parameter" in config, "parameter is required in config"
        self.seed, self.parameter, self.passing_reward_threshold = config["seed"], config["parameter"], config.get("passing_reward_threshold", 1.0)