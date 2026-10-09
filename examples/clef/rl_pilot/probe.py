"""Check native encoding and optionally measure pilot difficulty on step 2048."""

import json
import math
from collections import defaultdict
from pathlib import Path

import httpx
from tap import Tap
from transformers import AutoTokenizer

from examples.clef.data import encode_example, read_examples


class Args(Tap):
    data: Path
    model_path: Path
    endpoint: str = ''
    timeout: float = 10800


def probe(args: Args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    report = {}
    if args.timeout <= 0:
        raise ValueError('timeout must be positive')
    client = httpx.Client(timeout=args.timeout)
    if args.endpoint:
        health = client.get(args.endpoint+'/health')
        health.raise_for_status()
        assert health.json()['model_path'] == str(args.model_path)
    for split in ['train','validation']:
        examples = read_examples(args.data/f'{split}.jsonl')
        longest = 0
        metrics = defaultdict(lambda: {'n':0,'fields':0,'correct':0,'record_correct':0,'brier_sum':0.0,'collapsed':0})
        with (args.data/f'baseline-{split}.jsonl').open('w') as out:
            for index, example in enumerate(examples):
                encoded = encode_example(tokenizer,example,65536)
                longest = max(longest,len(encoded.encoded.input_ids))
                if args.endpoint:
                    response = client.post(args.endpoint+'/v1/systemone',json=example.record)
                    response.raise_for_status()
                    result = response.json()
                    probabilities = result['probabilities']
                    correct = []
                    losses = []
                    collapsed = 0
                    for field, target in example.targets.items():
                        predicted = probabilities[field]
                        assert set(predicted) == set(target)
                        assert all(math.isfinite(v) and v >= 0 for v in predicted.values())
                        assert abs(sum(predicted.values())-1) < 1e-5
                        answer = max(predicted,key=predicted.get)
                        correct.append(target[answer] == 1)
                        losses.append(sum((predicted[k]-v)**2 for k,v in target.items()))
                        collapsed += max(predicted.values()) >= 1-1e-6
                    m = metrics[example.source]
                    m['n'] += 1
                    m['fields'] += len(correct)
                    m['correct'] += sum(correct)
                    m['record_correct'] += all(correct)
                    m['brier_sum'] += sum(losses)/len(losses)
                    m['collapsed'] += collapsed
                    out.write(json.dumps({'id':example.record['id'],'source':example.source,'result':result})+'\n')
                if index % 128 == 0:
                    print('PROBE',split,index,flush=True)
        report[split] = {'records':len(examples),'longest_tokens':longest,'metrics':dict(metrics)}
    client.close()
    report['model_path'] = str(args.model_path)
    report['endpoint'] = args.endpoint
    (args.data/'native-probe.json').write_text(json.dumps(report,indent=2))
    print('PROBE_COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    probe(Args(underscores_to_dashes=True).parse_args())
