import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaxSegmentCoverageConstraint_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3602
    prompt_template = \
r"""You are given {N} segments (each is a closed interval [l, r]) on the x-axis:
{segments}

You are also given a list of constraints, where each constraint is a pair (p, x), meaning that the number of selected segments covering point p must be **at most** x:
{constraints}

Your task is to select the **maximum number of segments** (each can be selected at most once) such that all the constraints are satisfied. Output the indices of the selected segments in one line, separated by spaces."""

    def __init__(self,
                 coordinate_multiple : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxSegmentCoverageConstraint_Environment instance.
        """
        super().__init__(**kwargs)

        self.coordinate_multiple = coordinate_multiple
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

        coverages = [0] * (N * self.coordinate_multiple + 1)

        segments = self.parameter["segments"] = []
        for i in range(N) :
            l = random.randint(0, N * self.coordinate_multiple)
            r = random.randint(l, N * self.coordinate_multiple)
            segments.append((l, r))
            coverages[l] += 1
            if r + 1 < len(coverages) :
                coverages[r + 1] -= 1
        for p in range(1, len(coverages)) :
            coverages[p] += coverages[p - 1]
            assert coverages[p] >= 0, "Coverage should be non-negative"
        
        constraints = [p for p, coverage in enumerate(coverages) if coverage > 0]
        constraints = random.sample(constraints, random.randint(1, len(constraints)))
        constraints = self.parameter["constraints"] = [(p, random.randint(1, coverages[p])) for p in constraints]
        random.shuffle(constraints)


        # (3) make lists of exactly the needed length
        segments = segments.copy()
        points   = constraints.copy()
        
        # sort segments by left endpoint, but keep an ID for each
        segs = sorted([(l, r, idx) for idx, (l, r) in enumerate(segments)],
                    key=lambda x: x[0])
        # sort points by position
        pts = sorted(points, key=lambda x: x[0])
        
        # two heaps: min‐heap over (r, id), max‐heap over (-r, id)
        min_heap = []
        max_heap = []
        removed_ids = set()    # IDs of segments we've popped (expired or forcibly removed)
        size = 0               # current # of active segments covering p
        ans = N                # start assuming we keep all N
        i = 0                  # pointer into segs
        
        def clean_min():
            # drop any heap‐entries whose segment‐id is in removed_ids
            while min_heap and min_heap[0][1] in removed_ids:
                heapq.heappop(min_heap)
        
        def clean_max():
            while max_heap and max_heap[0][1] in removed_ids:
                heapq.heappop(max_heap)
        
        for p, x in pts:
            # 1) add every segment whose left ≤ p
            while i < N and segs[i][0] <= p:
                _, r, sid = segs[i]
                heapq.heappush(min_heap, (r, sid))
                heapq.heappush(max_heap, (-r, sid))
                size += 1
                i += 1
            
            # 2) expire any that end before p
            clean_min()
            while min_heap and min_heap[0][0] < p:
                r_exp, sid_exp = heapq.heappop(min_heap)
                size -= 1
                removed_ids.add(sid_exp)
                clean_min()
            
            # 3) if we exceed x overlap, remove the segments with the largest r
            clean_max()
            while size > x:
                neg_r, sid_rem = heapq.heappop(max_heap)
                size -= 1
                ans -= 1
                removed_ids.add(sid_rem)
                clean_max()
        
        assert ans > 0, "The answer should be greater than 0"
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            segments = "\n".join("Segment {}: [{}, {}]".format(i, l, r) for i, (l, r) in enumerate(self.parameter["segments"])),
            constraints = "\n".join("({}, {})".format(p, x) for p, x in self.parameter["constraints"]),
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

            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]

            coverages = [0] * (max(r for l, r in self.parameter["segments"]) + 1)
            for i in processed_result :
                l, r = self.parameter["segments"][i]
                coverages[l] += 1
                if r + 1 < len(coverages) :
                    coverages[r + 1] -= 1
            for p in range(1, len(coverages)) :
                coverages[p] += coverages[p - 1]
            
            for p, x in self.parameter["constraints"] :
                assert coverages[p] >= 0, "Coverage should be non-negative"
                if coverages[p] > x :
                    return self.rewards["invalid_solution"]
            
            answer, gold = len(processed_result), self.parameter["gold_answer"]
            assert answer <= gold, "The answer should be less than or equal to the gold answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]