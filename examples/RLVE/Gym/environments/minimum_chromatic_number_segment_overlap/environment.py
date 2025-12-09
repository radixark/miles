import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumChromaticNumber_SegmentOverlap_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2859
    prompt_template = \
r"""There are {N} segments (closed intervals) on the x-axis, labeled from `0` to `{N_minus_1}`:
{segments}

Your task is to assign a **non-negative integer color** to each segment, represented as `c[0], c[1], ..., c[{N_minus_1}]`, such that:
- If segment `u` and segment `v` overlap (i.e., they share at least one point), then `c[u] ≠ c[v]`.
- The total number of **distinct colors used** (i.e., unique values among `c[0]` to `c[{N_minus_1}]`) is **minimized**.

**Output Format:** A single line containing the color of each segment in order: `c[0] c[1] ... c[{N_minus_1}]` (separated by spaces).
Example: `0 1 0 2` means segment 0 has color 0, segment 1 has color 1, segment 2 has color 0, and segment 3 has color 2 (assuming 4 segments in total)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumChromaticNumber_SegmentOverlap_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        answer_upperbound = random.randint(2, N)
        segment_numbers = random.sample(range(1, N + 1), k = answer_upperbound - 1)
        segment_numbers.sort()
        segment_numbers += [N]
        for i in range(len(segment_numbers) - 1, 0, -1) :
            segment_numbers[i] -= segment_numbers[i - 1]
        
        segments = self.parameter["segments"] = []
        for segment_number in segment_numbers :
            endpoints = random.choices(range(1, 2 * N), k = 2 * segment_number)
            endpoints.sort()
            for i in range(0, len(endpoints), 2) :
                l = endpoints[i]
                r = endpoints[i + 1]
                segments.append((l, r))
        random.shuffle(segments)
        assert len(segments) == N, "len(segments) should be equal to N"
        

        segs = []
        for i, (a, b) in enumerate(segments):
            segs.append((a, b, i))  # (start, end, original_index)

        # Sort by start time
        segs.sort(key=lambda x: x[0])

        # Min-heap of (end_time, stall_id)
        heap = []
        next_stall_id = 0
        assignment = [0] * N  # assignment[i] = stall id for cow i (1-based ids)

        for l, r, idx in segs:
            if heap and heap[0][0] < l:
                # Reuse the earliest finishing stall
                _, stall_id = heapq.heappop(heap)
            else:
                # Need a new stall
                next_stall_id += 1
                stall_id = next_stall_id

            assignment[idx] = stall_id
            heapq.heappush(heap, (r, stall_id))

        self.parameter["gold_answer"] = next_stall_id
        self.parameter["reference_answer"] = " ".join(map(str, assignment))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            segments = "\n".join("Segment {}: [{}, {}]".format(i, l, r) for i, (l, r) in enumerate(self.parameter["segments"])),
        )


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

            colors = processed_result
            if len(colors) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            def overlap(seg1, seg2) -> bool :
                return max(seg1[0], seg2[0]) <= min(seg1[1], seg2[1])
            for u in range(self.parameter["N"]) :
                for v in range(u + 1, self.parameter["N"]) :
                    if overlap(self.parameter["segments"][u], self.parameter["segments"][v]) and colors[u] == colors[v] :
                        return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], len(set(colors))
            assert gold <= answer, "gold should be less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]