import math
import heapq
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class WYRLevelingGround_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3543
    prompt_template = \
r"""You are given an array H of {N} integers. Initially, it is: {H}
Your goal is to make every element in H equal to zero by applying a sequence of operations. A single operation is defined as choosing any non-empty contiguous subarray of H and applying one of the following four modifications to each element within that subarray:
- Add {A}
- Subtract {A}
- Add {B}
- Subtract {B}

Each time you apply one of these modifications to a subarray, it counts as one operation. What is the minimum total number of operations required to make all elements of H equal to zero?"""

    def __init__(self,
                 A_B_multiple : int = 2,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the WYRLevelingGround_Environment instance.
        """
        super().__init__(**kwargs)

        self.A_B_multiple = A_B_multiple
        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        while True :
            A, B = self.parameter["A"], self.parameter["B"] = random.randint(1, N * self.A_B_multiple), random.randint(1, N * self.A_B_multiple)
            if A != B :
                break
        
        positive_A_probability, positive_B_probability = random.random(), random.random()
        H = self.parameter["H"] = []
        for _ in range(N) :
            a_coeff, b_coeff = random.randint(0, N * self.A_B_multiple), random.randint(0, N * self.A_B_multiple)
            if random.random() < positive_A_probability :
                a_coeff = -a_coeff
            if random.random() < positive_B_probability :
                b_coeff = -b_coeff
            H.append(a_coeff * A + b_coeff * B)


        def extended_gcd(a, b):
            """
            Returns (x, y, g) such that a*x + b*y = g = gcd(a,b).
            """
            if b == 0:
                return 1, 0, a
            x1, y1, g = extended_gcd(b, a % b)
            # back-substitute
            return y1, x1 - (a // b) * y1, g

        def solve():
            # Build the difference array C of length N+1:
            # C[0] = H[0]; C[i] = H[i] - H[i-1] for i=1..N-1; C[N] = -H[N-1]
            new_N = N + 1
            C = [0] * new_N
            C[0] = H[0]
            for i in range(1, N):
                C[i] = H[i] - H[i-1]
            C[N] = -H[N-1]

            # Compute gcd and Bézout coefficients
            d = math.gcd(A, B)
            u, v, g = extended_gcd(A, B)
            # g == d
            ad = A // d
            bd = B // d

            # Prepare x[i], y[i] so that A*x[i] + B*y[i] = C[i], minimizing |x|+|y|
            x = [0] * new_N
            y = [0] * new_N
            dx = 0
            ans = 0
            sgn = lambda z: -1 if z < 0 else 1

            for i in range(new_N):
                ci = C[i]
                if ci % d != 0:
                    assert False, "C[i] should be divisible by d"

                factor = ci // d
                p0 = u * factor
                q0 = v * factor

                # Try the two shifts from the p0-based solution:
                best_x = p0 % bd
                best_y = (ci - A * best_x) // B
                best_cost = abs(best_x) + abs(best_y)

                # shift by one period in the x-direction
                cand_x = best_x - bd
                cand_y = best_y + ad
                cand_cost = abs(cand_x) + abs(cand_y)
                if cand_cost < best_cost:
                    best_x, best_y, best_cost = cand_x, cand_y, cand_cost

                # Now try the two shifts from the q0-based solution:
                alt_y = q0 % ad
                alt_x = (ci - B * alt_y) // A
                alt_cost = abs(alt_x) + abs(alt_y)
                if alt_cost < best_cost:
                    best_x, best_y, best_cost = alt_x, alt_y, alt_cost

                # one more shift
                cand_y2 = alt_y - ad
                cand_x2 = alt_x + bd
                cand_cost2 = abs(cand_x2) + abs(cand_y2)
                if cand_cost2 < best_cost:
                    best_x, best_y, best_cost = cand_x2, cand_y2, cand_cost2

                x[i] = best_x
                y[i] = best_y
                dx += best_x
                ans += best_cost

            # Build a min-heap of how much extra cost it costs to shift one unit of x (and compensate y)
            sign = sgn(dx)
            heap = []
            for i in range(new_N):
                nx = x[i] - sign * bd
                ny = y[i] + sign * ad
                delta = (abs(nx) + abs(ny)) - (abs(x[i]) + abs(y[i]))
                heapq.heappush(heap, (delta, i))

            # We need to do abs(dx)//bd such adjustments
            adjust_count = abs(dx) // bd
            for _ in range(adjust_count):
                delta, i = heapq.heappop(heap)
                ans += delta
                # apply the shift
                x[i] -= sign * bd
                y[i] += sign * ad
                # re-compute this index's next delta and re-push
                nx = x[i] - sign * bd
                ny = y[i] + sign * ad
                new_delta = (abs(nx) + abs(ny)) - (abs(x[i]) + abs(y[i]))
                heapq.heappush(heap, (new_delta, i))

            # Each boundary operation is counted twice, so divide by 2
            return ans // 2

        self.parameter["reference_answer"] = solve()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            H = " ".join("H[{}]={}".format(i, Hi) for i, Hi in enumerate(self.parameter["H"])),
            A = self.parameter["A"],
            B = self.parameter["B"],
        )
    

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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]