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

These scripts import their shared address book as `examples.…`, so run them as modules from the
repository root rather than by path:

```bash
SCRIPT=examples.infra_features.split_deployment.run_qwen3_0_6b_split

python -m $SCRIPT --deploy-component trainer
python -m $SCRIPT --deploy-component inference --deploy-instance-id e0
python -m $SCRIPT --deploy-component inference --deploy-instance-id e1
python -m $SCRIPT --deploy-component primary
```

When the run ends, the releases that carry no orchestration script are still up:

```bash
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-trainer -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-e0 -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-e1 -n $MILES_SCRIPT_NAMESPACE
```

## Example 2: A Run That Trains Several Policies

`run_solver_verifier_gsm8k_split.py` installs the [multi_policy](../../multi_policy) solver /
verifier demo the same way: one release per policy trainer, one per policy's engines, and the
orchestration script last — five releases for two policies.

Name this run once as well, so the example stands on its own:

```bash
export MILES_SCRIPT_RUN_ID=split-multi-policy-demo
export MILES_SCRIPT_RUN_UUID=$(python -c 'from miles.utils.run_uuid import generate_run_uuid; print(generate_run_uuid())')
```

```bash
SCRIPT=examples.infra_features.split_deployment.run_solver_verifier_gsm8k_split

python -m $SCRIPT --deploy-component trainer --deploy-instance-id solver-actor
python -m $SCRIPT --deploy-component trainer --deploy-instance-id verifier-actor
python -m $SCRIPT --deploy-component inference --deploy-instance-id solver
python -m $SCRIPT --deploy-component inference --deploy-instance-id verifier
python -m $SCRIPT --deploy-component primary
```

```bash
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-trainer-solver-actor -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-trainer-verifier-actor -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-solver -n $MILES_SCRIPT_NAMESPACE
helm uninstall miles-run-$MILES_SCRIPT_RUN_ID-inference-verifier -n $MILES_SCRIPT_NAMESPACE
```
