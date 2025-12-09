import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PrimeGraph_MinimumChromaticNumber_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `1` to `{N}`. Two vertices `u` and `v` are connected by an edge **if and only if** the absolute difference `|u - v|` is a prime number.

Your task is to assign a **non-negative integer color** to each vertex, represented as `c[1], c[2], ..., c[{N}]`, such that:
- For every edge `(u, v)` in the graph, `c[u] ≠ c[v]` — adjacent vertices must have different colors.
- The total number of **distinct colors used** (i.e., the number of unique values among `c[1]` to `c[{N}]`) is **minimized** - try your best to find a valid coloring using as few colors as possible.

**Output Format:** Your final answer should be a single line containing the color of each vertex in order: `c[1], c[2], ..., c[{N}]`, separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the PrimeGraph_MinimumChromaticNumber_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N)


        if N <= 6 :
            self.parameter["reference_answer"] = [(i + 1) // 2 for i in range(1, N + 1)]
        else :
            self.parameter["reference_answer"] = [i & 3 for i in range(1, N + 1)]
        self.parameter["gold_answer"] = len(set(self.parameter["reference_answer"]))
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            N = self.parameter["N"]

            colors = processed_result
            if len(colors) != N :
                return self.rewards["invalid_solution"]
            colors = [-1] + colors
            assert len(colors) == N + 1, "colors should be of length N + 1"
            
            is_prime = [True] * (N + 1)
            if N >= 0 :
                is_prime[0] = False
            if N >= 1 :
                is_prime[1] = False
            primes = []
            for i in range(2, N + 1) :
                if is_prime[i] :
                    primes.append(i)
                    for j in range(i * i, N + 1, i) :
                        is_prime[j] = False
            
            for p in primes :
                for i in range(1, N - p + 1) :
                    if colors[i] == colors[i + p] :
                        return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], len(set(colors[1 :]))
            assert gold <= answer, "gold should be less than or equal to answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]