import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class NoDoubleTripleCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3226
    prompt_template = r"""How many subsets of 1, 2, ..., {N} satisfy that if x is in the subset, then neither 2 × x nor 3 × x is in the subset?"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the NoDoubleTripleCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "N should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N)


        S = list(range(1, N + 1))
        assert len(S) == N, "S should contain exactly N elements"

        # visited[i] means “value i+1 has already been included in some component”
        visited = [False] * N

        def dp(root):
            # build the 2-chain: root, 2·root, 4·root, … ≤ n
            pow2_chain = []
            v = root
            while v <= N:
                pow2_chain.append(v)
                v *= 2
            L = len(pow2_chain)

            # for each of those, build its 3-chain: v, 3·v, 9·v, … ≤ n
            pow3_chains = []
            for v in pow2_chain:
                chain = []
                u = v
                while u <= N:
                    chain.append(u)
                    u *= 3
                pow3_chains.append(chain)

            # mark all nodes in this component
            for chain in pow3_chains:
                for u in chain:
                    visited[u - 1] = True

            # lmt0[i] = maximum mask value at level i (0…i=L)
            # level 0 has only mask 0
            lmt0 = [0] + [(1 << len(chain)) - 1 for chain in pow3_chains]

            # f[i][mask] = number of ways up to level i with configuration ‘mask’ at level i
            f = [[0] * (l + 1) for l in lmt0]
            f[0][0] = 1

            # transition from level i → i+1
            for i in range(L):
                for mask_j, ways in enumerate(f[i]):
                    if not ways:
                        continue
                    # try every subset mask_k on next 3-chain
                    for mask_k in range(lmt0[i + 1] + 1):
                        # no conflict with previous level, and no adjacent picks in this level
                        if (mask_j & mask_k) == 0 and (mask_k & (mask_k << 1)) == 0:
                            # f[i + 1][mask_k] = (f[i + 1][mask_k] + ways) % MOD
                            f[i + 1][mask_k] += ways

            # once you finish the last real level, all those mask-states are final
            # (in the original C++ they'd collapse through extra levels to mask 0,
            #  which is exactly summing f[L][*])
            # return sum(f[L]) % MOD
            return sum(f[L])

        ans = 1
        for x in S:
            if not visited[x - 1]:
                # ans = ans * dp(x) % MOD
                ans *= dp(x)
        
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])
    

    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]