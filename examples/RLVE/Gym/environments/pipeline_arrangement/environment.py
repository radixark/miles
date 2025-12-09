import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PipelineArrangement_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1248
    prompt_template = \
r"""You need to process {N} products labeled from `0` to `{N_minus_1}`. Each product must go through **two machines**, A and B, **in order**.

The processing times for each product on machines A and B are given as:
{A_and_B}

Please determine a permutation (i.e., an ordering) of all products. Each product is processed one by one in the chosen order:
- First on machine A.
- Then, after finishing on A, it waits (if needed) and is processed by machine B; meanwhile, machine A can continue processing subsequent products without any delay.
- Machine B processes one product at a time in the order they complete machine A.

Try your best to **minimize the time** when the **last product finishes** on machine B.

**Output Format:** Your final answer should be a single line containing the indices of the products in the chosen order (i.e., the permutation), separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the PipelineArrangement_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def get_finishing_time(self, order) -> int :
        tA = tB = 0
        for idx in order :
            tA += self.parameter["A"][idx]
            if tB < tA :
                tB = tA
            tB += self.parameter["B"][idx]
        return tB
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        A = self.parameter["A"] = [random.randint(1, N) for _ in range(N)]
        B = self.parameter["B"] = [random.randint(1, N) for _ in range(N)]


        tasks = []
        for i in range(N) :
            if A[i] < B[i] :
                tasks.append((A[i], 0, i))
            else:
                tasks.append((B[i], 1, i))

        tasks.sort(key = lambda x : x[0])

        order = [None] * N
        left, right = 0, N - 1
        for time, belong, idx in tasks :
            if belong == 0 :
                order[left] = idx
                left += 1
            else :
                order[right] = idx
                right -= 1

        self.parameter["reference_answer"] = " ".join(map(str, order))
        self.parameter["gold_answer"] = self.get_finishing_time(order)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A_and_B = "\n".join("A[{}]={}, B[{}]={}".format(i, self.parameter["A"][i], i, self.parameter["B"][i]) for i in range(N)),
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if len(set(processed_result)) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            
            answer, gold = self.get_finishing_time(processed_result), self.parameter["gold_answer"]
            assert gold <= answer

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]