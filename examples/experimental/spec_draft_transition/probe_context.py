"""Check cache rollback against an exactly enumerable, history-dependent teacher."""

from types import SimpleNamespace

import torch

from examples.experimental.spec_draft_transition.evaluate import sample_sequence


class HistoryCache:
    def __init__(self) -> None:
        self.tokens: list[int] = []

    def crop(self, length: int) -> None:
        assert length <= len(self.tokens)
        del self.tokens[length:]


class ContextModel:
    def __init__(self, table: torch.Tensor) -> None:
        self.table = table
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.model = self

    def parameters(self):
        yield self.parameter

    def __call__(self, input_ids: torch.Tensor, past_key_values: HistoryCache | None = None,
                 use_cache: bool = True) -> SimpleNamespace:
        assert input_ids.shape[0] == 1
        cache = past_key_values if past_key_values is not None else HistoryCache()
        logits = []
        for token in input_ids[0].tolist():
            cache.tokens.append(token)
            previous = cache.tokens[-2] if len(cache.tokens) > 1 else 0
            logits.append(self.table[3 * previous + token].log())
        return SimpleNamespace(logits=torch.stack(logits)[None], past_key_values=cache)


def verify_context() -> None:
    teacher_table = torch.tensor([[1 + (i * 3 + j * 5) % 7 for j in range(3)] for i in range(9)], dtype=torch.float32)
    student_table = torch.tensor([[1 + (i * 7 + j * 2 + 3) % 11 for j in range(3)] for i in range(9)], dtype=torch.float32)
    teacher_table /= teacher_table.sum(-1, keepdim=True)
    student_table /= student_table.sum(-1, keepdim=True)
    teacher, student = ContextModel(teacher_table), ContextModel(student_table)
    expected = torch.empty(3, 3, 3)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                expected[a, b, c] = teacher_table[2, a] * teacher_table[6 + a, b] * teacher_table[3 * a + b, c]
    observed = torch.zeros_like(expected)
    corrections = bonuses = 0
    torch.manual_seed(7941)
    for _ in range(4000):
        result = sample_sequence(teacher, student, [0, 2], 1.0, set(), 3, 3)
        a, b, c = result['generated_ids'][:3]
        observed[a, b, c] += 1
        corrections += result['corrections']
        bonuses += result['bonuses']
    error = float((observed / 4000 - expected).abs().max())
    assert error < 0.025, (error, observed / 4000, expected)
    assert corrections > 100 and bonuses > 100
    print({'contextual_three_token_maximum_error': error, 'corrections': corrections, 'bonuses': bonuses})


if __name__ == '__main__':
    verify_context()
