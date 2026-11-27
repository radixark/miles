---
name: top-add-family
description: Admit a new model (checkpoint or architecture) as a true on-policy (TOP) family, so its trainer log-probs equal its rollout log-probs bit for bit. Diff the model's forward against the op registry, add one paired Miles/SGLang commit per new op (serving program, trainer binding, strict cells, refusal tests), compose the family, audit what changes outside TOP, then add and run the family's e2e CI gate. Use when asked to add, port or qualify a model for --true-on-policy-mode.
---

# Add a TOP family

TOP (`--true-on-policy-mode`) makes the trainer's log-probs equal the rollout engine's bit for bit, by running one
batch-invariant numerical program on both sides. A model runs under TOP only if it belongs to an admitted **family**
in the op registry. A family is a list of **ops**, and each op declares the topology it supports (its **cells**).

This skill adds a family. It works in two checkouts at once: Miles (this repo) and the SGLang it pins. Steps 0-5 run
in order. Keep the ledger as you go: copy [ledger.md](ledger.md) to the run dir at step 0.

## Rules for every step

- Before the first edit, read every file in Miles' and SGLang's `.claude/rules/`.
- **One op = one numerical program.** If the new model needs a different branch, block size, rounding or kernel inside
  an existing op, that is a new op.
- **What a kernel must give TOP:** (a) a row's output bits do not depend on the other rows (batch size, composition);
  (b) prefill, chunked prefill, decode and verify compute the same bits for a token; (c) the trainer can call it on
  packed rows. SGLang's batch-invariant mode guarantees (a) only. A stock decode or verify kernel is often a different
  program from the prefill one, so check it on its own.
- **Kernel rules,** in order:
  1. reuse a registry op unchanged;
  2. reuse serving's kernel, made to meet (a)-(c) by configuration (pinned tiles and split counts, a fixed reduction
     order, no shape-dependent alternate branch; the existing ops' row comments show the patterns), or by a flag added
     to it that leaves its default output unchanged;
  3. prefer serving's fused kernel over an unfused chain. When the fused kernel rounds differently from the trainer's
     eager module, the trainer calls the fused kernel (backward case iii); serving never unfuses to copy the trainer's
     rounding;
  4. a new kernel only when 2 and 3 fail. It passes the same contract and has an unfused reference for its check. Both
     sides run it (serving dispatches it under TOP, the trainer delegates to it), and it stays TOP-only (step 3b).

  New ops have no perf target.
- **Say in the code why a stock path is not used.** A new kernel's docstring opens with `Stands in for <stock kernel>:
  <which of (a)-(c) it fails, and how>` (no measured numbers in code; they go in the PR body); speed with the same semantics is a reason if said so. A gate
  that turns a stock serving path off under TOP carries the same reason in a comment. "For exactness" alone is not a
  reason. Commit bodies do not survive a squash; the code does.
- **Precision selects ops, never cells** (`Family.fp8_ops`). Topology lives in cells; dtype, shape, backend and
  feature preconditions stay in the op's validator.
- **Cells are strict.** A cell opens only with a run that qualifies it, recorded in the ledger and named in the docs
  row. Every other degree is refused with a mechanism or the literal `NOT_QUALIFIED`. The CI gate runs a
  representative subset of the admitted cells, not all of them.
- **Fail closed.** Whatever the qualification does not cover is refused by name, never left to a generic path.
- **Tests:** each op commit adds refusal tests: CPU unit tests that the op raises on what it does not support,
  instead of silently running another program (e.g. TOP sampling raises on `top_p`; the TOP RMSNorm raises when the
  weight dtype differs from the input's). Numerical parity is checked by the family's e2e CI gate, since offline tests
  cannot enumerate runtime dispatch. The exception is a stateful op's path the gate's slice cannot reach (a verify
  window with rejected drafts, a rollback, a chunk close): give it a fast GPU kernel test, seconds not minutes, that
  compares the path bitwise with the plain program (the KDA and GDN row tests under
  `test/registered/e2e/true_on_policy/` in SGLang show the shape).
- **A check you write yourself proves nothing until you have seen it red** (step 4).

## When to stop and ask the user

| Stop | Why |
|---|---|
| Step 0 fails | porting the model itself is not TOP work |
| An op another family uses would change its program | that family's gate must rerun |
| A hunk outside TOP is ungated and was not intended | it changes behaviour for every user |
| A choice the checkpoint and the request do not settle (e.g. which weight format to qualify) | the user's call |
| **Any GPU run** | show the commands, node and estimated GPU-hours, and wait; one ask may cover the whole gate |

Record each question and answer in the ledger. Decide everything else yourself and record why.

---

## The op registry

Read both halves before step 1. Their docstrings and row comments are the reference.

| Half | File | Owns |
|---|---|---|
| SGLang | `python/sglang/srt/true_on_policy/registry.py` | `Cell` and helpers (`EVERY`, `only`, `at_least`, `refused`), `Spec`, `ServingEnvelope`, `Op`, `Family`, `OPS`, `FAMILIES`, the matcher, `admit_serving` |
| Miles | `miles_plugins/top/registry.py` | `TrainerEnvelope`, `TRAINER_CELLS`, `TopBuild`, `BUILDERS`, the core trainer refusals, `admit_trainer` |

- **Op:** one numerical operation with a trainer/serving contract (the BI GEMM, an attention program, an expert
  program, a draft head). SGLang owns the list; Miles keys its tables by the same names.
- **Axis:** a parallel degree an op supports or refuses. Serving: `ep`, `dp_attention`, `pp`, `spec` (the
  `(steps, topk, draft_tokens)` recipe). Trainer: `ep`, `cp` (all-gather degree), `pp`. TP and expert TP are not axes:
  a core rule fixes them at 1 on the trainer, and serving never splits a reduction.
- **Cell:** what one op says about one axis. Every refused degree carries a reason: a mechanism, or `NOT_QUALIFIED`.
  Never a run name or a SHA. The struct refuses to build a restricting cell without a reason.
- **Envelope:** one complete set of cells for one side. An op may list several, to state cross-axis rules.
- **Family:** model types (trainer and serving spellings) plus the ops their forward runs; `fp8_ops` lists the ops a
  block-FP8 checkpoint runs instead, and `vision_ops` the ops a run with images or video adds.
- **Stateful ops:** an op whose output depends on per-request state built over earlier tokens (a recurrent or
  convolution state) must give prefill, decode and verify one program. Read the existing GDN and KDA rows and their
  serving programs before you add one.
- **Conventions in `OPS`:** each row has a comment stating its program (kernel, dtypes, rounding, what is pinned). New
  ops start from `_STRICT_SERVING` (no serving stages, no speculation); `_SPEC_SERVING` is for ops qualified under
  speculation. A draft op is serving-only (`in_trainer=False`), and its `spec` cell lists its qualified recipes. A
  family admits speculation only when every one of its ops does.
- **Where admission runs:** Miles at argument parsing (`miles/backends/megatron_utils/arguments.py` calls
  `admit_trainer`, so a refused layout fails before Ray starts); Miles when the block is built
  (`miles_plugins/top/spec.py`, then `BUILDERS[family.name]`); SGLang at engine launch
  (`python/sglang/srt/entrypoints/engine.py` calls `admit_serving`). Each side checks its own half; never derive the
  serving layout in Miles.
- **Existing guards:** Miles `tests/fast/miles_plugins/top/test_registry.py` fails if an op or family lacks its trainer
  half. SGLang `test/registered/unit/true_on_policy/test_registry.py` covers the matcher. Do not add mirror tests of
  cell contents.

---

## Step 0. Preconditions

- Read the checkpoint config (a nested `text_config` included), both `model_type` spellings and `quantization_config`.
- Serving support must exist in SGLang (`python/sglang/srt/models/`), and trainer support in Miles: the Megatron
  model or bridge, HF-to-Megatron conversion and the `megatron_to_hf` export. If either is missing, stop.
- Both trees must have the registry (the two files above).
- Copy [ledger.md](ledger.md) to the run dir and fill in its header.

## Step 1. Diff the forward against `OPS`

Output: one table in the ledger, one row per module or surface.

| Module / surface | Verdict | Existing op or new name | Evidence (file:line on both sides) |
|---|---|---|---|

Verdicts are **reuse** (the program is unchanged), **new op**, or **refused** (by mechanism, with the refusal's
owner). Do not stop for review after this step.

1. **Pick the nearest admitted family** in `FAMILIES` (same lineage, same serving model file) and start from its op
   tuple. Read its builder (`BUILDERS`) and its e2e gate to see how its ops are bound and qualified.
2. **Derive the layer schedule from the config:** every layer kind and per-layer variant, including layer-indexed
   lists, relations where one layer consumes another's output or cache, and layers whose kind differs from their
   neighbours. A module that looks shared can run different branches on different layers.
3. **Read the serving model file and the trainer model side by side,** module by module, for each layer kind. Follow
   each call into the kernel it dispatches, and note the branch conditions (config values, shapes, env flags,
   backends, prefill versus decode). Follow the decode and verify calls too: they often reach another kernel.
4. **Reuse an op only if its program is unchanged:** the same kernel, block and group sizes, rounding, branch and
   dtypes.
5. **Add rows beyond the forward:**
   - checkpoint quantization, and how the trainer's masters relate to the bytes serving runs (load, sync, export);
   - speculative drafts in scope: one row per draft method, with its draft op and its verify program under TOP. Drafts
     serve the checkpoint's draft weights; TOP does not train them. The draft op samples with the same sampler key
     the verify step uses at that position;
   - state carried across decode steps or across layers;
   - checkpoint load and weight export, including weights that are not trained parameters;
   - features the qualification refuses, each with its mechanism.
6. **Backward reference:** pick the first route that applies; it decides the gate's backward phase (step 4).
   1. **The vendor's own reference code** (the model's official inference file), when the release ships one.
      `miles/utils/debug_utils/vendor_reference.py` imports it with its kernel imports served by PyTorch FP32
      stand-ins, so it runs under autograd. The vendored file must be byte-diffable against the release.
   2. **HF transformers,** when the vendor ships no reference and HF implements this `model_type` in the run image.
   3. **Op-level FP32 references**, one per new op, when neither exists.

   A reference that still runs quantized kernels is not an FP32 autograd reference.
7. **Gate vehicle:** an N-layer real-weight slice whose layers cover every op and every layer kind of the schedule,
   with source and consumer layers both present. Write down how it is cut.

## Step 2. One paired commit per new op

Commit each op as soon as its CPU tests pass, before starting the next one; never let several ops' changes pile up
uncommitted. Each op is one commit in each repo. Subjects are one line,
`feat(top): <op> op: <what serving and trainer share>`. The commits that added the nearest family's ops show the shape
(`git log -- python/sglang/srt/true_on_policy/registry.py` in SGLang).

**SGLang commit:**
- the serving program under `python/sglang/srt/true_on_policy/`, which raises instead of falling back to another
  program;
- the `Op(...)` row, with its program comment and strict serving cells;
- the op's entry in `true_on_policy/kernel_manifest.py` (the kernels that decide its bits, their pin sites, the stock
  fast paths it turns off), then `kernel_inventory.py generate` and `update` in `true_on_policy/`;
  `test_kernel_inventory.py` fails on an op without an entry or a stale inventory;
- refusal tests in `test/registered/unit/true_on_policy/`, each naming its failure mode (`unit-test-admission.md`).

**Miles commit:**
- the trainer binding under `miles_plugins/top/`: the forward delegates to the SGLang op on the same inputs; the
  backward follows "Wiring the backward" below;
- the builder binds it through `binding.bind_roles`, which records the role on the build stamp;
- dump points: the op's input and output under one name on both sides (`binding.canon_dump(name, tensor)` in Miles,
  `dumper.dump(name, tensor, layer_id=...)` from `sglang.srt.debug_utils.dumper` in SGLang), so a mismatch can be
  bracketed later without new code;
- the `TRAINER_CELLS` row;
- refusal tests under `tests/fast/miles_plugins/top/`.

**Refusal baseline.** The core launch refusals already cover every family (`miles/true_on_policy/args.py`,
`device.py`): LoRA, external engines and engine-contract overrides, a critic, attention dropout, an entropy loss,
greedy or transformed sampling (rollout and eval), and a trainer and engines on different GPU architectures. Do not
re-add them. Per family, the trainer config refuses what the TOP program would silently ignore; the existing families'
checks are the baseline (`miles_plugins/top/moe_ep.py`, `glm_config.py`, `dsv4_spec.py`):
- a router that routes as serving does: router aux loss, z-loss, forced load balancing, `--moe-router-dtype`;
- experts bound to another expert module: `--moe-grouped-gemm`;
- a layout rule serving enforces (e.g. GLM's index sharing): checked with serving's own function.

Each gets its refusal test.

**Topology checks move into cells.** If a validator of a touched op checks EP, CP, PP or a spec recipe, move the check
into the op's cells and delete it from the validator. Validators keep dtype, shape, backend and feature rules.

**Non-forward rows** from step 1 (master handling, export names, untrained weights) get their own commits in the same
shape, with refusal tests for what they do not support.

### Wiring the backward

Decide per module, and write the choice into the op's row in the ledger:

| Case | When | How |
|---|---|---|
| (i) framework backward | the delegated forward is the same math the framework's module differentiates | keep the module's autograd; only the forward call changes |
| (ii) straight-through | the forward runs on quantized bytes of a trained master | the forward returns the quantized program's output; the backward is evaluated at the served weight (the dequantization of the deployed bytes), and its weight gradient goes straight through to the master (`miles_plugins/top/fp8_forward.py` `_StraightThrough` is the pattern). Evaluating it at the master instead differentiates a different network wherever the master sits off the served value (an off-grid offset, a coarse format), and the contract check fails it (step 4) |
| (iii) explicit Function | neither applies | an autograd Function with an analytic VJP, or a surrogate: the forward value is serving's, the gradient that of a PyTorch reference of the same math recomputed at the saved inputs (`miles_plugins/top/kernels/surrogate.py`: `surrogate`, `linear_surrogate`, `row_surrogate`, `reference_mode`) |

Rules:
- One autograd Function per module.
- Return the hard forward's bits. Never `soft + (hard - soft).detach()`: the sum does not always round back to `hard`.
- No in-place or raw kernel in the training graph without a VJP.
- Recompute equals the scored forward: activation recompute replays the same delegated program, never a reference
  forward, because backward differentiates the recompute while the log-probs came from the scored forward.
  `miles_plugins/top/install.py` `refuse_unpatched_backward` lists the recompute paths TOP refuses.
- dX and dW are evaluated at the served weights; the contract check (step 4) replays each module there.
- A contract table per op in the ledger: the dtype on each side (FP32 main parameter, BF16 master, deployed bytes),
  and where each conversion happens.

## Step 3. Compose the family

One commit per repo, `feat(top): admit <Family>, composed of <ops>, at <layout>`:
- SGLang `FAMILIES`: `Family(name=…, model_types=(<trainer spelling>, <serving spelling>), ops=…, fp8_ops=…)`;
- Miles `BUILDERS[name] = TopBuild(build="module:function", server_kwargs="module:function")`: the block builder, and
  the `ServerArgs` keywords of the serving program the trainer pins;
- `miles/true_on_policy/launch.py`: the family's flags and env in `top_launch`;
- the model's launcher script under `scripts/`: a `--true-on-policy` option that uses `top_launch`;
- the "Admitted families" row in `examples/infra_features/true_on_policy/README.md` (a pre-commit hook mirrors it into
  `docs/`): family, `model_type`, admitted layouts, gate file, and the admitted cells the gate does not run (as the
  existing rows do);
- the family's `FamilyAdapter` in `miles/utils/debug_utils/top_reference.py` `ADAPTERS`, for backward routes 1 and 2
  (step 4).

Open only the cells a recorded run qualifies (trainer EP/CP/PP, serving EP, DP-attention width, PP, spec recipes).

**CPU proof, before any GPU:**
- every commit passes its own tests;
- Miles `tests/fast/miles_plugins` and `tests/fast/utils/test_arguments.py`; SGLang
  `test/registered/unit/true_on_policy`;
- an import smoke of every touched module;
- an admission check: call `admit_trainer` and the SGLang `admit` with the gate's layouts (admitted) and their
  neighbours (refused, with the expected message).

Record each run and its pass/fail counts in the ledger.

## Step 3b. Outside-TOP audit

After wiring, before any gate. It proves the family's hooks into shared files are inert with TOP off; it is the first
thing reviewers ask.

1. **List every hunk outside the TOP modules:** SGLang `python/sglang/srt/true_on_policy/`; Miles `miles_plugins/top/`,
   `miles/true_on_policy/`; tests.
2. **Name each hunk's gate with TOP off:** the TOP contract or switch (`is_true_on_policy_enabled`,
   `resolve_true_on_policy_runtime_policy(...).enabled`, `publish_role() is not None and ...`,
   `--true-on-policy-mode`) or a default-off argument. A model-file edit counts as gated only if what it changed acted
   under TOP alone. A hunk with no gate is a shared change: say what it does for everyone, and stop if it was not
   intended.
3. **Check the spill paths:**
   - new keywords passed to shared kernels: does every callee (FA3 included) accept or drop them?
   - new hot-path reads of the TOP switch, which raise without published server args (does the base already read it
     there?);
   - new module-level imports in shared files, which load on every platform (keep Triton/CUDA TOP modules lazy);
   - shared buffer dtypes (logits stay FP32 with TOP off);
   - stricter kernel paths stay TOP-only.
4. **List the kernel choices:** every kernel the family adds and every stock serving path it turns off under TOP, each
   with the reason the code gives (Rules: "Say in the code why"). SGLang's `true_on_policy/kernel_inventory.md` lists
   them per op, and its "Not classified" section names those without a reason. A missing reason is fixed before the
   gate: either write it, or switch to the stock path.
5. **Write the PR description's "Outside TOP" section:** what changes with TOP off, not a work log. Copy it into the
   ledger.

## Step 4. The e2e gate, as CI

**The gate file:** `tests/e2e/precision/test_<model>_<N>layer_true_on_policy.py`. Read the existing
`tests/e2e/precision/test_*_true_on_policy.py` gates first and follow their shape:
- a docstring that states the vehicle, every phase and what each proves;
- `register_cuda_ci(...)` (see `tests/ci/ci_register.py` and `ci-test-registration.md`);
- the slice in a public Hugging Face repo, downloaded by the gate, since CI and outside users fetch it there;
- `--ci-test` with `--true-on-policy-mode`: `train_rollout_logprob_abs_diff` must be 0 at each rollout's first
  optimizer step, the one that scores the weights that sampled it (later steps score updated weights). Run at least
  two rollouts, so one scores weights after an optimizer step and the weight sync is covered;
- `top_launch` and `TOP_RECIPE_TRAIN_ARGS` from `miles/true_on_policy/launch.py`;
- phases for the main layouts (each phase names the cells it qualifies);
- a speculative phase if the family admits speculation;
- a backward phase.

**Backward phase.** The log-prob check cannot see a wrong gradient: both sides then agree on the wrongly updated
weights. The existing gates show the wiring (`top_capture.capture_args`, `top_vjp.vjp_verdict`, `top_capture.fail_on`);
the module docstrings of `top_capture`, `top_layout`, `top_vjp` and `top_reference` are the reference.
- **Capture:** every phase captures the same decoder layers: at least one of each layer kind, and layers on both sides
  of a PP stage boundary when a phase runs PP.
- **Same rollouts:** phases compared with each other must have trained the same step-1 rollouts; this is checked before
  any cross-phase comparison.
- **Layout** (`top_layout`): every phase against the `cp1` phase, per captured tensor (parameter gradients, layer input
  gradients, output cotangents, the worst token row and the worst expert); routing and index selections bitwise.
- **Contract** (`top_vjp`, judged): is the backward implemented as declared? Each module in the adapter's `contract` is
  replayed in FP32 in the reference at TOP's captured module inputs, at the served weights, with TOP's routing and
  index selections and the deployed KV rounding straight-through. Its forward must reproduce TOP's output, and TOP's
  gradients must sit within the module's measured floor times `CONTRACT_FACTOR`, inside the cosine and norm-ratio
  bands. A backward taken at other weights than the forward ran fails here.
- **Quality** (`top_vjp`, reported only): each layer against an unquantized FP32 reference, beside HF BF16's own
  distance from it. It mixes the declared approximation with rounding, so it has no threshold; record it.
- **Adapter:** the family's `FamilyAdapter` in `miles/utils/debug_utils/top_reference.py` `ADAPTERS` (its fields are
  documented there): the reference loader, the layer run, the parameter-name map, the `contract` modules, the
  selection replay, `kv_quantization`, and `token_terms` for gradients that are sums over tokens (norm scales, mixing
  scalars), which can nearly cancel and are then judged against the sum of their per-token magnitudes and by sign.
  For route 1, `load` is the vendor module's loader and `config` is `vendor_reference.read_config_json`; the
  "DeepSeek-V4" adapter and `top_reference_dsv4.py` are the worked example.
- **Floors:** a new family has none, and a module without a measured floor fails. Measure them with
  `MILES_TOP_GATE_REPORT_ONLY=1` on known-good captures that vary what drives the error, not only the seed: at least
  three rollout seeds and a second engine width or batch composition, so routing and expert load differ between
  captures. The floor is the largest value; record its spread (largest over median) beside it, and give a module whose
  spread already nears `CONTRACT_FACTOR` more captures before the floor is set. Add each constant to `top_vjp.py` /
  `top_layout.py` with the captures it came from, then rerun enforced. Before trusting them, show the verdict failing
  on planted mutations of a saved capture (a flipped sign, a swapped expert, a one-index routing change). A floor
  measured against a wrong reference is too loose: re-measure a family's floors whenever its reference changes, and
  log the old and new values.
- **Contract bias, optional:** where the masters differ from the served weights, `masters_path` reports how far a
  backward at the masters would be; a number in the ledger.
- **Route 3:** an op-level backward of each new op against an independent FP32 autograd reference of the same math,
  on inputs captured from the gate run. Write it in the run dir, not the repo. Fix the tolerance from the op's math and
  write it down before the run. Record the result in the ledger.

**Before calling it green,** check in the log and record in the ledger:
- every step scored the samples and tokens the batch should give (a diff of 0 over nothing proves nothing);
- the run's layout and spec recipe are the ones the phase names (read the resolved server args and trainer config);
- the build stamp lists the new ops' bindings, so the TOP path, not a generic one, ran;
- no `[top]` "paths that no TOP gate runs" warning (`miles/true_on_policy/unverified.py`) in any phase's log.

**Admitted but not gated.** A path the family admits that no gate runs is listed: in `unverified_paths` when a launch
can detect it, otherwise in the "Not yet verified" section of `examples/infra_features/true_on_policy/README.md`.

**Checks you write yourself** (an op-level backward reference, a comparison script): before trusting a pass, show it
failing on a planted one-ulp or one-index error. The CI gate's own log-prob checker needs no planted run.

**Running it:** GPU only with the user's OK. Show the commands, the node and the estimated GPU-hours.

**A nonzero diff.** Use the repo's tools; do not write your own dump or compare scripts.
1. **Freeze the failing step.** Rerun the phase with `--dump-details <dir>`: it saves the rollout
   (`<dir>/rollout_data/<rollout_id>.pt`: tokens, rollout log-probs, seeds) and the trainer's side
   (`<dir>/train_data/`). From then on, `--load-debug-rollout-data <dir>/rollout_data/{rollout_id}.pt` rescores the
   same tokens on the trainer alone, with no engines, so each trainer-side bisect step is cheap.
2. **Find the tokens.** The gate metric is a per-sample mean. Read the per-token difference from the saved rollout
   and train data (`miles/utils/debug_utils/display_debug_rollout_data.py` reads the rollout file): which samples,
   which positions, the first differing position, and whether it is prefill or decode.
3. **Capture both sides at the same names.** Add `--dumper-dir <dir>` with `--dumper-inference enable=true
   filter='layer_id=[...]'` (serving) and `--dumper-fwd-only enable=true filter=...` (trainer scoring);
   `--dumper-fwd-bwd` captures the training forward and gradients (`top_vjp.capture_args` shows the pattern).
   `non_intrusive_mode=all` also hooks every module; the default `core` keeps only the explicit dump points the op
   commits added. Keys are `DumperConfig` fields in `sglang/srt/debug_utils/dumper.py`. Each phase writes its own
   subdirectory (`inference/`, `fwd_only/`, `fwd_bwd/`).
4. **Compare.** `python -m sglang.srt.debug_utils.comparator --baseline-path <dir>/inference --target-path
   <dir>/fwd_only --preset sglang_megatron` aligns serving's steps with the trainer's packed rows; `--filter` narrows
   to names, `--visualize-per-token <png>` draws per-token differences. The culprit is the first op whose inputs are
   equal and whose output differs.
5. **Prove it is the op, not its input.** The dumper's grafter copies serving's input into the trainer at that point:
   `grafter_enable=true`, `grafter_role=baseline` on serving and `target` on the trainer, `grafter_b2t_filter` naming
   the op's input, plus the master address, port and world sizes `DumperConfig` asks for. If the output still differs,
   the op's program differs; if not, look upstream.
6. **Read both sides' resolved branch and kernel** at that op: server args as published to the trainer, env flags,
   shapes, the build stamp.
7. **Change one factor at a time** until the difference goes away and comes back. Fix it in the commit that owns the
   op, rerun the CPU proof, and record the cause in the ledger.

When it passes, fill the gate column of the docs table row.

## Step 5. Perf sanity (no threshold)

- TOP against native on the same engine layout and vehicle: rollout tokens/s and train step time.
- Speculative families: accept length, TOP against native, on the same weights and temperature.
- Profile both (Miles `--profile-target`; SGLang's profiler). Compare the top kernels by time. Flag unfused or
  fallback paths in the TOP profile (a PyTorch reference op, a non-batch-invariant GEMM) and their share of step time.
- Report the rows and flags in the ledger. No threshold: this step finds accidental slow paths, not targets.

## Done

- Every new op and the family are committed in both trees in the shape above, and each commit passes its own tests.
- The "Outside TOP" section is written.
- The e2e gate passes on GPU with the user's OK, with the green checks above recorded.
- The docs table row names the admitted cells, the gate, and which admitted cells the gate does not run; admitted
  paths no gate runs are listed as above. The ledger holds the op table, the contract tables, every decision and question,
  and the perf rows.
