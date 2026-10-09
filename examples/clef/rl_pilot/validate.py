"""Recompute pilot labels and execute exact field/record reward checks."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tap import Tap

from generate import make_case


class Args(Tap):
    data: Path


def score(example: dict[str, Any], actions: dict[str, str]) -> dict[str, float]:
    expected = example['metadata']['ground_truth']['answers']
    if set(actions) != set(expected):
        raise ValueError('missing or unknown decision fields')
    for key, value in actions.items():
        if value not in example['record']['questions'][key]['criteria']:
            raise ValueError('unknown option')
    correct = [float(actions[k] == value) for k, value in expected.items()]
    return {'field_accuracy':sum(correct)/len(correct), 'record_success':float(all(correct))}


def validate(args: Args) -> None:
    manifest = json.loads((args.data/'manifest.json').read_text())
    groups: dict[str, set[str]] = {}
    report: dict[str, Any] = {}
    for split in ['train','validation']:
        path = args.data/f'{split}.jsonl'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest['sha256'][split]
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == manifest[split]
        groups[split] = set()
        counts: Counter[str] = Counter()
        fields = 0
        for row in rows:
            truth = row['metadata']['ground_truth']
            index = int(truth['id'].split('-')[-1])
            rebuilt = make_case(index,split,manifest['seed'])
            assert rebuilt == truth
            assert row['record']['questions'] == truth['questions']
            assert truth['policy'] in row['record']['state']
            assert all(fact in row['record']['state'] for fact in truth['facts'])
            assert set(row['targets']) == set(truth['answers'])
            for field, answer in truth['answers'].items():
                target = row['targets'][field]
                assert set(target) == set(truth['questions'][field]['criteria'])
                assert all(value == float(key == answer) for key,value in target.items())
            assert score(row,truth['answers']) == {'field_accuracy':1.0,'record_success':1.0}
            wrong = {k:next(v for v in truth['questions'][k]['criteria'] if v != answer) for k,answer in truth['answers'].items()}
            assert score(row,wrong) == {'field_accuracy':0.0,'record_success':0.0}
            try:
                score(row,{})
            except ValueError:
                pass
            else:
                raise AssertionError('incomplete answer was accepted')
            audit = row['metadata']['audit']
            assert audit['answers'] == {k:truth['questions'][k]['criteria'][v] for k,v in truth['answers'].items()}
            assert audit['unambiguous'] and not audit['unsupported_claims']
            groups[split].add(hashlib.sha256(row['record']['state'].encode()).hexdigest())
            counts[truth['family']] += 1
            fields += len(truth['answers'])
        assert len(groups[split]) == len(rows)
        report[split] = {'records':len(rows),'fields':fields,'families':dict(counts),'sha256':digest}
    assert not groups['train'] & groups['validation']
    report['checks'] = ['deterministic ground-truth regeneration','complete evidence preservation',
                        'native choice target alignment','exact correct/incorrect reward probes',
                        'incomplete answer rejection','unique states within/across splits','blind reviewer agreement']
    (args.data/'validation-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    validate(Args().parse_args())
