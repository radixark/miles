import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class KosDicing_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3425
    prompt_template = \
r"""There are {N} players (labeled from 0 to {N_minus_1}) participating in a game consisting of {M} rounds. Each round (a, b) involves two distinct players a and b, given as:
{rounds}

In each round, exactly one of the two players wins. Please determine the outcome of all rounds such that the **maximum number of total wins by any player** is exactly {K} (basically, each player has a number of wins, and the maximum of these numbers is exactly {K}).

**Output Format:** Output {M} integers, separated by spaces. The i-th integer represents the winner of the i-th round, either a or b (do NOT include backticks or quotes)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, correct_solution : float = +1.0, wrong_solution : float = 0.0,
                 **kwargs) :
        """
        Initialize the KosDicing_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "correct_solution": correct_solution,
            "wrong_solution": wrong_solution,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"

        rounds = self.parameter["rounds"] = []
        reference_answer = []
        winning_counts = [0] * N
        for _ in range(M) :
            a, b = random.sample(range(N), 2)
            rounds.append((a, b))
            winner = random.choice((a, b))
            winning_counts[winner] += 1
            reference_answer.append(winner)
        assert len(rounds) == M, "The number of rounds should be exactly M"
        self.parameter["K"] = max(winning_counts)
        self.parameter["reference_answer"] = " ".join(map(str, reference_answer))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            M = self.parameter["M"],
            rounds = "\n".join("({}, {})".format(a, b) for a, b in self.parameter["rounds"]),
            K = self.parameter["K"],
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["M"] :
                return self.rewards["invalid_solution"]
        
            counting = [0] * self.parameter["N"]
            for players, winner in zip(self.parameter["rounds"], processed_result) :
                if winner not in players :
                    return self.rewards["invalid_solution"]
                counting[winner] += 1
            if max(counting) != self.parameter["K"] :
                return self.rewards["wrong_solution"]
            else :
                return self.rewards["correct_solution"]
        else :
            return self.rewards["wrong_format"]