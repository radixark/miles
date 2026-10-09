"""Generate auditable decision cases; API credentials never enter artifacts."""

import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tap import Tap


class Args(Tap):
    output: Path
    key_file: Path = Path('/home/ubuntu/openai.key')
    count: int = 2048
    validation_count: int = 256
    concurrency: int = 24
    seed: int = 261009


def field(instructions: str, options: list[str]) -> dict[str, Any]:
    return {'type': 'choice', 'instructions': instructions,
            'criteria': {str(i): v for i, v in enumerate(options)}}


def make_case(index: int, split: str, seed: int) -> dict[str, Any]:
    rng = random.Random(f'{seed}:{split}:{index}')
    family = ['invoice', 'support', 'security', 'agent', 'tool', 'retrieval', 'extract', 'tool'][index % 8]
    # Quotas: workflows50%, tool25%, retrieval12.5%, extraction12.5%.
    facts: list[str] = []
    questions: dict[str, Any] = {}
    answers: dict[str, str] = {}

    def ask(name: str, instruction: str, options: list[str], correct: str) -> None:
        rng.shuffle(options)
        questions[name] = field(instruction, options)
        answers[name] = next(k for k, v in questions[name]['criteria'].items() if v == correct)

    ref = f'{split.upper()}-{rng.randrange(10000000,99999999)}'
    if family == 'invoice':
        ordered = rng.randrange(4, 81)
        received = rng.choice([ordered, ordered, ordered-rng.randrange(1, ordered)])
        price = rng.randrange(11, 501)
        invoice = ordered*price + rng.choice([0, 0, rng.randrange(1, 101)])
        vendor = rng.choice(['verified', 'verified', 'unverified'])
        duplicate = rng.choice([False, False, True])
        facts = [f'Purchase order {ref}: {ordered} units at {price} USD per unit.',
                 f'Warehouse receipt for {ref}: {received} units received.',
                 f'Invoice for {ref}: total {invoice} USD, tax and shipping both zero.',
                 f'Vendor verification status is {vendor}.',
                 f'Exact invoice previously paid: {str(duplicate).lower()}.',
                 f'An unrelated order {ref}-OLD has been received and paid in full.']
        policy = ('Reject an already-paid invoice. Otherwise hold an unverified vendor, incomplete delivery, '
                  'or a total unequal to ordered units times unit price. Approve only if none applies. '
                  'Escalation priority: duplicate to accounts payable; unverified vendor to security; '
                  'incomplete delivery to procurement; amount mismatch to accounts payable; otherwise none.')
        action = 'reject' if duplicate else 'hold' if vendor != 'verified' or received != ordered or invoice != ordered*price else 'approve'
        escalation = 'accounts payable' if duplicate else 'security' if vendor != 'verified' else 'procurement' if received != ordered else 'accounts payable' if invoice != ordered*price else 'none'
        ask('action', 'Choose the invoice action under the policy.', ['approve','hold','reject'], action)
        ask('escalation', 'Choose the escalation destination under the priority rule.', ['accounts payable','security','procurement','none'], escalation)
        ask('delivery', 'Is this purchase order fully received?', ['yes','no'], 'yes' if received == ordered else 'no')
        ask('amount', 'Does the invoice total match the purchase order?', ['yes','no'], 'yes' if invoice == ordered*price else 'no')
    elif family == 'support':
        days = rng.randrange(1, 61)
        opened = rng.choice([True, False])
        damaged = rng.choice([True, False])
        proof = rng.choice([True, False])
        facts = [f'Ticket {ref}: purchase occurred {days} days ago.', f'Package opened: {opened}.',
                 f'Customer reports product arrived damaged: {damaged}.', f'Proof of purchase provided: {proof}.',
                 'A different customer was refunded yesterday; that decision has no bearing on this ticket.']
        policy = ('If proof is missing, request proof before deciding. With proof, damaged arrivals qualify '
                  'for replacement within 45 days. Otherwise unopened purchases qualify for refund within '
                  '30 days. All other cases are denied. Damaged cases route to quality; others to billing. '
                  'Only request a return shipment for an approved refund or replacement.')
        action = 'request proof' if not proof else 'replace' if damaged and days <= 45 else 'refund' if not opened and days <= 30 else 'deny'
        ask('action','Choose the next support action.', ['request proof','replace','refund','deny'], action)
        ask('route','Choose the destination team.', ['quality','billing'], 'quality' if damaged else 'billing')
        ask('return','Request a return shipment now?', ['yes','no'], 'yes' if action in ['replace','refund'] else 'no')
    elif family == 'security':
        failures = rng.randrange(0, 15)
        mfa = rng.choice([True,False])
        approved = rng.choice([True,False])
        privilege = rng.choice([True,False])
        facts = [f'Account {ref}: {failures} failed logins in the last hour.', f'Latest successful login passed MFA: {mfa}.',
                 f'New device is on the approved device list: {approved}.', f'Privilege elevation requested: {privilege}.',
                 'A scheduled vulnerability scan is running on a different host.']
        policy = ('Block if MFA failed or at least 10 failed logins occurred. Otherwise review if the device '
                  'is unapproved or privilege elevation is requested; otherwise allow. Escalate to incident '
                  'response for block, identity team for review, and nobody for allow. Never grant privilege '
                  'elevation unless the action is allow.')
        action = 'block' if not mfa or failures >= 10 else 'review' if not approved or privilege else 'allow'
        ask('action','Select the access disposition.', ['block','review','allow'], action)
        ask('route','Select the escalation team.', ['incident response','identity team','nobody'], {'block':'incident response','review':'identity team','allow':'nobody'}[action])
        ask('elevation','Grant privilege elevation now?', ['yes','no'], 'yes' if privilege and action == 'allow' else 'no')
    elif family == 'agent':
        consent = rng.choice([True, False])
        verified = rng.choice([True, False])
        balance = rng.randrange(20, 900)
        amount = rng.randrange(10, 1000)
        facts = [f'Transfer request {ref}: {amount} USD.', f'Available balance: {balance} USD.',
                 f'Identity verified: {verified}.', f'Explicit authorization for this transfer: {consent}.',
                 'The customer authorized a separate transfer last month.']
        policy = ('First verify identity if unverified; next request authorization if absent; next reject if '
                  'amount exceeds balance; otherwise execute. Do not debit unless executing. Route identity '
                  'verification to identity, authorization to customer, insufficient funds to billing, '
                  'and execution to payments.')
        action = 'verify identity' if not verified else 'request authorization' if not consent else 'reject' if amount > balance else 'execute'
        ask('action','Choose the next action.', ['verify identity','request authorization','reject','execute'], action)
        ask('route','Choose the responsible team or party.', ['identity','customer','billing','payments'], {'verify identity':'identity','request authorization':'customer','reject':'billing','execute':'payments'}[action])
        ask('debit','Debit the account now?', ['yes','no'], 'yes' if action == 'execute' else 'no')
    elif family == 'tool':
        operation = rng.choice(['read invoice','cancel subscription','change address','send receipt'])
        has_id = rng.choice([True,False])
        auth = rng.choice([True,False])
        cached = rng.choice([True,False])
        definitions = {
            'invoice_lookup': 'Read an invoice; requires customer ID; no authorization required.',
            'subscription_cancel': 'Cancel a subscription; requires customer ID and explicit authorization.',
            'address_update': 'Change account address; requires customer ID and explicit authorization.',
            'receipt_send': 'Send a receipt; requires customer ID and explicit authorization.',
            'invoice_search': 'Search public sample invoices only; cannot access customer invoices.',
            'subscription_status': 'Read subscription status only; cannot cancel.',
        }
        wanted = {'read invoice':'invoice_lookup','cancel subscription':'subscription_cancel','change address':'address_update','send receipt':'receipt_send'}[operation]
        facts = [f'Request {ref}: please {operation}.', f'Customer ID supplied: {has_id}.',
                 f'Explicit authorization for the requested operation supplied: {auth}.',
                 f'Current invoice is already present in the conversation: {cached}.',
                 'Tool catalog: '+json.dumps(definitions, sort_keys=True)]
        policy = ('For reading an invoice already present, answer without a tool. Otherwise request missing '
                  'customer ID before any tool call; then request authorization for mutations if missing. '
                  'If all required inputs exist, select exactly the capable tool. Do not call sample-data '
                  'or status-only tools to perform customer mutations.')
        tool = 'no tool: answer from context' if operation == 'read invoice' and cached else 'no tool: ask customer ID' if not has_id else 'no tool: ask authorization' if operation != 'read invoice' and not auth else wanted
        ask('next','Choose the next action or tool.', list(definitions)+['no tool: answer from context','no tool: ask customer ID','no tool: ask authorization'], tool)
        ask('call','Should a tool be called immediately?', ['yes','no'], 'yes' if tool in definitions else 'no')
    elif family == 'retrieval':
        version = rng.randrange(2, 20)
        tier = rng.choice(['enterprise','consumer','education'])
        region = rng.choice(['west','east','north','south'])
        window = rng.choice([7,14,21,30,45])
        names = ['D1','D2','D3','D4','D5']
        rng.shuffle(names)
        good = names[0]
        docs = {good:f'Policy version {version}, {tier} plan, {region} region: refund window is {window} days.',
                names[1]:f'Policy version {version-1}, {tier} plan, {region} region: refund window is {window+7} days.',
                names[2]:f'Policy version {version}, other plan, {region} region: refund window is {window+14} days.',
                names[3]:f'Policy version {version}, {tier} plan, other region: refund window is {window+21} days.',
                names[4]:'Glossary: refund policies vary by plan, region and version; no numeric window is given.'}
        facts = [f'Query {ref}: under policy version {version}, what is the refund window for the {tier} plan in the {region} region?']+[f'{k}: {v}' for k,v in sorted(docs.items())]
        policy = 'Use only a document matching version, plan and region exactly. Superseded or differently scoped policies are ineligible.'
        ask('document','Which document directly answers the query?', names.copy(), good)
        ask('window','What is the applicable refund window in days?', [str(x) for x in [window,window+7,window+14,window+21]], str(window))
    else:
        owner = rng.choice(['Mira Chen','Daniel Ortiz','Samira Patel','Owen Blake'])
        deadline = rng.choice(['Monday','Tuesday','Wednesday','Thursday','Friday'])
        confirmed = rng.choice([True,False])
        complete = rng.choice([True,False])
        facts = [f'Project {ref}: owner is {owner}.', f'Proposed delivery day: {deadline}.',
                 f'Delivery proposal formally confirmed: {confirmed}.', f'Completion receipt issued: {complete}.',
                 'An older unrelated project was completed on Sunday by Alex Morgan.']
        policy = ('Extract only this project. Report delivery day as unknown unless formally confirmed. '
                  'Classify status as completed if a completion receipt exists; otherwise scheduled if '
                  'delivery is confirmed; otherwise pending. Do not infer confirmation from a proposal.')
        ask('owner','Who owns this project?', ['Mira Chen','Daniel Ortiz','Samira Patel','Owen Blake'], owner)
        ask('delivery','What is the confirmed delivery day?', ['Monday','Tuesday','Wednesday','Thursday','Friday','unknown'], deadline if confirmed else 'unknown')
        ask('status','What is the project status?', ['completed','scheduled','pending'], 'completed' if complete else 'scheduled' if confirmed else 'pending')
    return {'id': f'rl-pilot-{split}-{index:05d}', 'family': family, 'facts': facts,
            'policy': policy, 'questions': questions, 'answers': answers,
            'split': split, 'scenario_seed': f'{seed}:{split}:{index}'}


async def api_json(client: AsyncOpenAI, system: str, content: str) -> tuple[dict[str, Any], dict[str, Any]]:
    response = await client.chat.completions.create(
        model='gpt-6-luna', reasoning_effort='none', max_completion_tokens=2400,
        response_format={'type':'json_object'},
        messages=[{'role':'system','content':system}, {'role':'user','content':content}],
    )
    if response.choices[0].finish_reason != 'stop':
        raise ValueError('incomplete response')
    return json.loads(response.choices[0].message.content), {'response_id':response.id,
            'model':response.model, 'usage':response.usage.model_dump()}


async def generate_one(client: AsyncOpenAI, case: dict[str, Any], root: Path) -> None:
    path = root/'accepted'/f"{case['id']}.json"
    if path.exists():
        return
    traces: list[dict[str, Any]] = []
    for attempt in range(4):
        try:
            system = ('Write a realistic business evidence packet, not an answer. Return JSON {"documents": '
                      '[{"title":string,"text":string}]}. Use every supplied fact placeholder [[F0]], [[F1]], etc '
                      'exactly once in document text. We will replace them with exact facts. Add natural '
                      'email/memo/record framing and transitions, but NO additional factual claims, numerical '
                      'values, policy rules, decisions, hints, or answers. Divide facts among 2-4 related '
                      'documents. Do not include the policy in documents. Vary style, order and titles.')
            payload = {'family':case['family'], 'facts':{f'[[F{i}]]':f for i,f in enumerate(case['facts'])},
                       'style': ['email chain','operations memo','case notes','document bundle'][int(case['id'].split('-')[-1])%4]}
            rendered, usage = await api_json(client, system, json.dumps(payload))
            traces.append({'phase':'render','attempt':attempt,'usage':usage,'output':rendered})
            docs = rendered['documents']
            if not 2 <= len(docs) <= 4:
                raise ValueError('document count')
            joined = '\n'.join(d['text'] for d in docs)
            expected = [f'[[F{i}]]' for i in range(len(case['facts']))]
            if sorted(re.findall(r'\[\[F\d+\]\]',joined)) != sorted(expected):
                raise ValueError('fact placeholder coverage')
            for d in docs:
                for i, fact in enumerate(case['facts']):
                    d['text'] = d['text'].replace(f'[[F{i}]]',fact)
            state = 'Applicable policy (authoritative):\n'+case['policy']+'\n\nEvidence packet:\n'+ '\n\n'.join(d['title']+'\n'+d['text'] for d in docs)
            record = {'id':case['id'], 'state':state, 'questions':case['questions']}
            audit, audit_usage = await api_json(client,
                'Independently solve every requested decision using only the policy and evidence. '
                'Return JSON {"answers":{field_id:option_id},"unambiguous":boolean,"unsupported_claims":boolean,'
                '"explanation":string}. Set unsupported_claims true if framing adds decision-relevant '
                'facts not grounded in the supplied canonical fact list. Option IDs are strings. Do not '
                'use outside policies or assumptions.',
                json.dumps({'record':record,'canonical_facts':case['facts']}))
            traces.append({'phase':'audit','attempt':attempt,'usage':audit_usage,'output':audit})
            if audit['answers'] != case['answers'] or audit['unambiguous'] is not True or audit['unsupported_claims'] is not False:
                raise ValueError('independent audit disagreement')
            targets = {k:{o:float(o == case['answers'][k]) for o in v['criteria']} for k,v in case['questions'].items()}
            example = {'record':record,'targets':targets,'source':'clef_rl_pilot_'+case['family'],
                       'metadata':{'split':case['split'],'scenario_seed':case['scenario_seed'],
                                   'generator':'gpt-6-luna','checker_version':'1',
                                   'ground_truth':case,'audit':audit,'api_traces':traces}}
            path.write_text(json.dumps(example)+'\n')
            return
        except (ValueError, KeyError, TypeError) as error:
            traces.append({'phase':'rejection','attempt':attempt,'type':type(error).__name__,'reason':str(error)})
        except Exception as error:
            # API error text may include credential-related content; retain only type.
            traces.append({'phase':'api_error','attempt':attempt,'type':type(error).__name__})
            await asyncio.sleep(min(2**attempt,8))
    (root/'rejected'/f"{case['id']}.json").write_text(json.dumps({'case':case,'traces':traces}))


def finalize(args: Args) -> None:
    accepted = sorted((args.output/'accepted').glob('*.json'))
    rows = [json.loads(p.read_text()) for p in accepted]
    for split, count in [('train',args.count),('validation',args.validation_count)]:
        subset = [r for r in rows if r['metadata']['split'] == split]
        if len(subset) != count:
            raise ValueError(f'{split}: accepted {len(subset)} of {count}; inspect rejected cases')
        random.Random(f'{args.seed}:{split}:shuffle').shuffle(subset)
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subset))
    ids = [r['record']['id'] for r in rows]
    states = [hashlib.sha256(r['record']['state'].encode()).hexdigest() for r in rows]
    assert len(set(ids)) == len(rows) == len(set(states))
    manifest = {'seed':args.seed,'model':'gpt-6-luna','train':args.count,'validation':args.validation_count,
                'families':dict(Counter(r['source'] for r in rows)),
                'validation':'canonical facts inserted verbatim; blind API solve matches deterministic labels',
                'limitations':['Same-model independent reviewer, not independent human verification.',
                               'Fixed rule families shared across splits; distinct scenario seeds and IDs.',
                               'No external benchmark material used; semantic decontamination not proven.',
                               'Model-error mining and executable tool sandbox evaluation not yet performed.'],
                'sha256':{s:hashlib.sha256((args.output/f'{s}.jsonl').read_bytes()).hexdigest() for s in ['train','validation']}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE',json.dumps(manifest),flush=True)


async def main(args: Args) -> None:
    for name in ['accepted','rejected']:
        (args.output/name).mkdir(parents=True,exist_ok=True)
    client = AsyncOpenAI(api_key=args.key_file.read_text().strip(),max_retries=3,timeout=90)
    semaphore = asyncio.Semaphore(args.concurrency)
    done = 0

    async def worker(case: dict[str, Any]) -> None:
        nonlocal done
        async with semaphore:
            await generate_one(client,case,args.output)
            done += 1
            if done % 32 == 0:
                print('PROGRESS',done,'accepted',len(list((args.output/'accepted').glob('*.json'))),flush=True)

    cases = [make_case(i,split,args.seed) for split,n in [('train',args.count),('validation',args.validation_count)] for i in range(n)]
    await asyncio.gather(*(worker(c) for c in cases))
    await client.close()
    finalize(args)


if __name__ == '__main__':
    asyncio.run(main(Args(underscores_to_dashes=True).parse_args()))
