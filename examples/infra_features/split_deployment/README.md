# Split Deployment Example

One run installed by hand as one helm release per component: one per trainer, one per inference
engine, and one carrying the orchestration script.

## Example 1: A Single Policy Run

Inside a miles workbench the cluster backend and the namespace are already set; name the run once:

```bash
export MILES_SCRIPT_RUN_ID=split-demo
export MILES_SCRIPT_RUN_UUID=$(python -c 'from miles.utils.run_uuid import generate_run_uuid; print(generate_run_uuid())')
```

Then install one component per command — identical but for `--deploy-component` and
`--deploy-instance-id`, with `primary` last since installing it blocks until the run ends:

```bash
SCRIPT=examples/infra_features/split_deployment/run_qwen3_0_6b_split.py

python $SCRIPT --deploy-component trainer
python $SCRIPT --deploy-component inference --deploy-instance-id e0
python $SCRIPT --deploy-component inference --deploy-instance-id e1
python $SCRIPT --deploy-component primary
```

When the run ends, the releases that carry no orchestration script are still up:

```bash
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-trainer -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-e0 -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-e1 -n $MILES_SCRIPT_NAMESPACE
```

Uninstalling by hand races the primary release's own uninstall job, which fires two minutes after
the verdict. Whichever runs second finds the release already gone, and `--ignore-not-found` makes
that a successful uninstall rather than a failed job.

## Example 2: A Run That Trains Several Policies

`run_solver_verifier_gsm8k_split.py` installs the [multi_policy](../../multi_policy) solver /
verifier demo the same way: one release per policy trainer, one per policy's engines, and the
orchestration script last — five releases for two policies.

Each trainer release carries only its named policy, but remains policy-scoped:

- Training metrics and process identities keep the policy's model ID.
- `--save`, `--load`, and `--save-hf` resolve under `trainers/<trainer-id>`, so policy releases never share a checkpoint writer.

```bash
SCRIPT=examples/infra_features/split_deployment/run_solver_verifier_gsm8k_split.py

python $SCRIPT --deploy-component trainer --deploy-instance-id solver-actor
python $SCRIPT --deploy-component trainer --deploy-instance-id verifier-actor
python $SCRIPT --deploy-component inference --deploy-instance-id solver
python $SCRIPT --deploy-component inference --deploy-instance-id verifier
python $SCRIPT --deploy-component primary
```

```bash
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-trainer-solver-actor -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-trainer-verifier-actor -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-solver -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-verifier -n $MILES_SCRIPT_NAMESPACE
```
