# Loss Snapshot Tests

Regression tests that verify loss function outputs are bitwise identical to saved snapshots.

## Quick Start

```bash
# 1. Save snapshots from current code
python -m pytest tests/fast/backends/training_utils/loss/test_loss_snapshot.py --snapshot -v

# 2. Make your changes to loss code

# 3. Compare against snapshots
python -m pytest tests/fast/backends/training_utils/loss/test_loss_snapshot.py --compare -v
```

## Adding a new config

Edit `CONFIGS` in `test_loss_snapshot.py`:

```python
CONFIGS = [
    ...
    # (name, args_overrides, batch_size, prompt_lens, response_lens)
    ("grpo_opsm_b2",
     dict(advantage_estimator="grpo", loss_type="policy_loss", use_opsm=True),
     2, [40, 60], [20, 40]),
]
```

Then re-run `--snapshot`.
