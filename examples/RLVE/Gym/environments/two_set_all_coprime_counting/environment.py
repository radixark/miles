import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TwoSet_AllCoprime_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2150
    prompt_template = \
r"""You are given a set of integers: {set}

Please compute the number of set pairs (S, T) such that:
1. S and T are disjoint subsets of the given set.
2. For every x in S and y in T, gcd(x, y) = 1 (i.e., there is no pair with gcd > 1)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the TwoSet_AllCoprime_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        set_size = random.randint(2, N - 1)
        A = self.parameter["set"] = random.sample(range(2, N + 1), set_size)

        assert len(A) == len(set(A)) == set_size, "The set must contain unique integers"


        MAX = max(A)

        is_prime = [True] * (MAX + 1)
        is_prime[0] = is_prime[1] = False
        max_prime_factor = [None] * (MAX + 1)
        for i in range(2, MAX + 1) :
            if is_prime[i] :
                max_prime_factor[i] = i
                for j in range(2 * i, MAX + 1, i) :
                    is_prime[j] = False
                    max_prime_factor[j] = i

        group2numbers = {}
        small_primes = dict()
        for a in A :
            prime_factors = []
            x = a
            while x > 1 :
                prime = max_prime_factor[x]
                prime_factors.append(prime)
                x //= prime
            
            assert max(prime_factors) == prime_factors[0], "The largest prime factor must be the first one"
            if prime_factors[0] * prime_factors[0] > MAX :
                group = prime_factors[0]
                prime_factors = [prime for prime in prime_factors if prime != group]
                if group not in group2numbers :
                    group2numbers[group] = []
                group2numbers[group].append(prime_factors)
            else :
                group2numbers[-a] = [prime_factors]
            
            for prime in prime_factors :
                if prime not in small_primes :
                    small_primes[prime] = len(small_primes)
        F = [[0] * (1 << len(small_primes)) for S in range(1 << len(small_primes))]
        F[0][0] = 1
        for group, prime_factors_list in group2numbers.items() :
            G0 = [[F[S][T] for T in range(1 << len(small_primes))] for S in range(1 << len(small_primes))]
            G1 = [[F[S][T] for T in range(1 << len(small_primes))] for S in range(1 << len(small_primes))]
            for prime_factors in prime_factors_list :
                mask = 0
                for prime in prime_factors :
                    mask |= (1 << small_primes[prime])
                
                new_G0 = [[G0[S][T] for T in range(1 << len(small_primes))] for S in range(1 << len(small_primes))]
                new_G1 = [[G1[S][T] for T in range(1 << len(small_primes))] for S in range(1 << len(small_primes))]
                for S in range(1 << len(small_primes)) :
                    T = (1 << len(small_primes)) - 1 - S
                    while True :
                        assert (T & S) == 0, "S and T must be disjoint"
                        if (mask & T) == 0 :
                            new_G0[S | mask][T] += G0[S][T]
                        if (mask & S) == 0 :
                            new_G1[S][T | mask] += G1[S][T]
                        if T == 0 :
                            break
                        T = (T - 1) & ((1 << len(small_primes)) - 1 - S)
                G0 = new_G0
                G1 = new_G1
            for S in range(1 << len(small_primes)) :
                for T in range(1 << len(small_primes)) :
                    F[S][T] = G0[S][T] + G1[S][T] - F[S][T]
        
        self.parameter["reference_answer"] = sum(F[S][T] for S in range(1 << len(small_primes)) for T in range(1 << len(small_primes)))
        assert self.parameter["reference_answer"] > 0
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(set = " ".join(map(str, self.parameter["set"])))
    

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