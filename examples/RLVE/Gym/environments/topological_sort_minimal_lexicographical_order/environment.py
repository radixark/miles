import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TopologicalSort_MinimalLexicographicalOrder_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3243
    prompt_template = \
r"""Please find a permutation of `0` to `{N_minus_1}` ({N} integers in total) such that the following conditions are satisfied:
{before_conditions}

If multiple permutations satisfy the conditions, choose the one where:
(1) `0` should appear as early as possible;
(2) Subject to that, `1` should appear as early as possible;
(3) Subject to that, `2` should appear as early as possible;
(4) And so on...

**Output Format:** Your final answer should be a single line containing the permutation `p(0), p(1), ..., p({N_minus_1})`, separated by spaces."""

    def __init__(self,
                 max_indeg : int = 3, # Maximum in-degree of each vertex
                 wrong_format : float = -1.0, invalid_solution : float = -0.5,
                 rewarding_strategy_toposort : str = "(satisfied/all)^beta", rewarding_weight_toposort : float = +0.5, rewarding_beta_toposort : float = 5.0,
                 rewarding_strategy_lexicographical : str = "mean([gold=answer])^beta", rewarding_weight_lexicographical : float = +0.5, rewarding_beta_lexicographical : float = 5.0,
                 **kwargs) :
        """
        Initialize the TopologicalSort_MinimalLexicographicalOrder_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_indeg = max_indeg
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy_toposort" : rewarding_strategy_toposort,
            "rewarding_weight_toposort" : rewarding_weight_toposort,
            "rewarding_beta_toposort" : rewarding_beta_toposort,
            "rewarding_strategy_lexicographical" : rewarding_strategy_lexicographical,
            "rewarding_weight_lexicographical" : rewarding_weight_lexicographical,
            "rewarding_beta_lexicographical" : rewarding_beta_lexicographical,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 2"

        permutation = list(range(N))
        random.shuffle(permutation)

        before_conditions = self.parameter["before_conditions"] = []
        while True :
            for i in range(N) :
                if i == 0 :
                    continue
                for j in random.sample(range(i), random.randint(0, min(i, self.max_indeg))) :
                    before_conditions.append((permutation[j], permutation[i]))
            if before_conditions :
                break
        random.shuffle(before_conditions)


        # --- build the reverse graph (Y → X) --------------------------------
        adjacency = [[] for _ in range(N)]       # adjacency[u] holds every v with edge u→v
        indeg      = [0] * N                     # in-degree of each vertex

        for before, after in before_conditions:
            adjacency[after].append(before)
            indeg[before] += 1

        # --- Kahn’s algorithm with a *max*-heap ------------------------------
        pq = []
        for i in range(N):
            if indeg[i] == 0:
                heapq.heappush(pq, -i)           # negate to turn min-heap into max-heap

        order = []                               # extraction order
        while pq:
            u = -heapq.heappop(pq)               # restore original index
            order.append(u)
            for v in adjacency[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(pq, -v)

        # --- output ----------------------------------------------------------
        if len(order) < N:                       # a cycle exists
            assert False
        else:
            self.parameter["gold_answer"] = list(reversed(order))  # store the gold answer as a list of integers
            self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            before_conditions = "\n".join("{} must be before {}".format(j, i) for j, i in self.parameter["before_conditions"]),
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

            permutation = processed_result
            if len(permutation) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if len(set(permutation)) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in permutation) :
                return self.rewards["invalid_solution"]
            
            positions = [None] * self.parameter["N"]
            for i, p in enumerate(permutation) :
                positions[p] = i


            reward = 0.0

            satisfied = sum(positions[j] < positions[i] for j, i in self.parameter["before_conditions"])
            assert satisfied <= len(self.parameter["before_conditions"]), "satisfied should not exceed the number of conditions"
            if self.rewards["rewarding_strategy_toposort"] == "(satisfied/all)^beta" :
                reward += self.rewards["rewarding_weight_toposort"] * ((satisfied / len(self.parameter["before_conditions"])) ** self.rewards["rewarding_beta_toposort"])
            elif self.rewards["rewarding_strategy_toposort"] == "satisfied/all" :
                reward += self.rewards["rewarding_weight_toposort"] * (satisfied == len(self.parameter["before_conditions"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_toposort"]))
            
            if satisfied == len(self.parameter["before_conditions"]) :
                if self.rewards["rewarding_strategy_lexicographical"] == "mean([gold=answer])^beta" :
                    reward += self.rewards["rewarding_weight_lexicographical"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], permutation)) / self.parameter["N"]) ** self.rewards["rewarding_beta_lexicographical"])
                elif self.rewards["rewarding_strategy_lexicographical"] == "gold=answer" :
                    reward += self.rewards["rewarding_weight_lexicographical"] * (self.parameter["gold_answer"] == permutation)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_lexicographical"]))

            return reward
        else :
            return self.rewards["wrong_format"]