"""Exercise reporting on paired synthetic measurements with known differences."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from examples.experimental.spec_draft_transition.analyze import main


def verify_analysis() -> None:
    samples, curves, jobs = [], [], []
    for temperature in [0.6, 1.0, 1.4]:
        for index, method in enumerate(['ce', 'l1', 'smooth']):
            for seed in range(20001, 20011):
                common = {'run_id': f'{temperature}-{method}-{seed}',
                          'temperature': temperature, 'method': method, 'seed': seed,
                          'learning_rate': 3e-5}
                jobs.append({**common, 'wandb_url': 'fixture'})
                blocks = np.random.default_rng(seed).integers(1, 33, size=2000)
                for prompt, count in enumerate(blocks):
                    samples.append({**common, 'draft': 8, 'prompt_id': f'{prompt:08d}',
                                    'blocks': int(count),
                                    'accepted': int(count) * (2 + index + prompt % 2 + seed % 2)})
                for update in [0, 500, 1000, 2000]:
                    curves.append({**common, 'update': update,
                                   'overlap': 0.5 + index * 0.05 + update * 0.0001,
                                   'kl': 0.4 - index * 0.05 - update * 0.0001})
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pl.DataFrame(samples).write_parquet(root / 'samples-node0.parquet')
        pl.DataFrame(curves).write_parquet(root / 'curves-node0.parquet')
        pl.DataFrame(jobs).write_parquet(root / 'jobs-node0.parquet')
        (root / 'state.json').write_text(json.dumps({'source_revision': 'fixture'}))
        main(SimpleNamespace(root=str(root), resamples=512, seed=837))
        result = json.loads((root / 'summary.json').read_text())
        for row in result['primary_contrasts']:
            expected = 2.0 if row['baseline'] == 'ce' else 1.0
            for key in ['difference', 'lower_adjusted', 'upper_adjusted']:
                assert abs(row[key] - expected) < 1e-10, row
        for row in result['secondary_interactions']:
            assert abs(row['high_minus_low_gain']) < 1e-10, row
        for name in ['learning-curves.png', 'paired-curves.png', 'accepted-prefix.png']:
            assert (root / name).stat().st_size > 1000, name
    print('Paired analysis fixture passed: exact +2/+1 contrasts and zero entropy interaction.')


if __name__ == '__main__':
    verify_analysis()
