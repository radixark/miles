# Sample ownership analysis

Check each eligible source sample for one complete outcome on every current model replica, or one explicit whole-source drop.

## Reading route

| File | Read for |
| --- | --- |
| [check.py](check.py) | Entry point `check`, eligibility, per-source/per-replica resolution |
| [issued.py](issued.py) | Retry deduplication and conflicting GRPO slot identities |
| [witness.py](witness.py) | Current cohort selection and complete output-set validation |
| [models.py](models.py) | Identity, witness, and resolution issue types |
| [analyzer.py](../../analyzer.py) | Startup grace and raising when the rule returns issues |
| [test_check.py](../../../../../../tests/fast/utils/event_analyzer/rules/sample_ownership/test_check.py) | Executable examples: `TestMaturity`, `TestPerSlotAccounting`, `TestCompactRows`, `TestExplicitDrops`, `TestCurrentTrainerWitnesses` |

## Evidence reaches the rule

1. A participating Megatron rank publishes its CPU witness after successful actor training and returns its unique `snapshot_id` through `TrainStepOutput`.
2. The existing `TrainGroupStepEndEvent` records cell outcomes and `sample_ownership_snapshot_ids` returned by that attempt.
3. [store.py](../../../sample_ownership/store.py) selects accepted `NORMAL` cells and matches snapshots by exact ID, rollout, and attempt. Completion is not inferred from cross-host timestamp comparisons.
4. The store combines issuance/drop history with those snapshots and a synthesized `TrainerWitnessCohortEvent`.
5. [checker.py](../../../sample_ownership/checker.py) reads published evidence and invokes the analyzer; no controller witness-collection RPC is needed.

- **Maturity cutoff**: Ranks publish their step-window cutoff; the store uses the most conservative replica cutoff. `check` receives the cutoff through `now` and `grace_period`.
- **Rule boundary**: `check(events, grace_period=..., now=...)` only reads its supplied event list and returns structured issues.

## Identities and inputs

| Input | Meaning |
| --- | --- |
| `DataSourceIssuedSamplesEvent` | Each `(group_index, slot, sample_index)` is an obligation; `slot` is its position in the group's `sample_indices` |
| [`SampleLineage`](../../../../types.py) | `(source_sample_index, output_index, output_count)` identifies one output and its declared sibling count |
| `TrainerCpuWitnessEvent` | Per-replica current counts, separated into trained and nonfinite-skipped rows; `TrainingSampleCount` carries the lineage as `SampleLineagePayload` |
| `TrainerWitnessCohortEvent` | Names the completed cohort, rollout, and expected replicas |
| `ExplicitlyDroppedSamplesEvent` | Each appearance in `sample_indices` counts as one whole-source drop |

- **Join key**: Issuance's `sample_index` joins witness rows through `source_sample_index`; a GRPO group total is insufficient.
- **Snapshot meaning**: Counts describe current model state. Historical snapshots and replicated copies must not be added together.

## Algorithm

### 1. Normalize issuance

- Deduplicate identical `(group_index, slot, sample_index)` tuples and retain the earliest timestamp; retrying cannot refresh age.
- If one sample index names multiple `(group_index, slot)` pairs, return `IssuedSampleIdentityIssue` and exclude that ambiguous index from resolution checks.

### 2. Select the current witness cohort

- Select the latest marker by `(timestamp, input position)`, then the latest matching snapshot for each expected replica by the same ordering.
- Require matching `cohort_id` and `rollout_id`; ignore unlisted replicas and older cohorts.
- Missing marker, empty replica set, or missing snapshots produce `CurrentTrainerWitnessIssue`; stop resolution checking but retain identity issues.
- The highest rollout number is not necessarily current: restoring a checkpoint can lower it. Historic successful consumption cannot repair a missing outcome in current state.

### 3. Select eligible sources

Group trained and skipped rows by source. Check the union of:

- Valid issued sources whose age satisfies `now - issued_at >= grace_period`.
- Sources observed in any selected replica's trained or skipped witness rows, including young sources and sources without issuance evidence.

Boundary cases:

- **Every call**: Recheck the entire eligible set; previous success never removes an obligation.
- **`now=None`**: Only observed sources qualify; negative grace raises `ValueError`.
- **Missing issuance**: Check observed sources with `group_index=None` and `slot=None`.
- **Drops alone**: Do not trigger eligibility; young unobserved sources wait for grace, and unissued/unobserved indices are outside the selected set.

### 4. Check each eligible source on every replica

Let `D` be its total number of explicit-drop appearances, including duplicates inside one payload.

| Condition | Requirement | Failure |
| --- | --- | --- |
| `D > 1` | Invalid | One source-level issue with `replica_id=None` |
| `D == 1` | No trained or skipped rows on any selected replica | An issue for each conflicting replica |
| `D == 0` | One complete output set on every selected replica | An issue for each incomplete or duplicated replica outcome |

For `D == 0`, concatenate trained and skipped rows for that source and replica. Require:

1. A nonempty set of rows with one common positive `output_count`, called `N`.
2. Exactly `N` rows, with indices exactly `0` through `N - 1`.
3. Every row's `count` equals `1`.

- Different siblings may be trained or nonfinite-skipped, but the same sibling cannot appear in both outcomes.
- Every replica must independently pass; a complete peer cannot hide another replica's missing row.
- A whole-source drop cannot coexist with even one trained or skipped sibling.

## Examples

Assume mature issuance and a valid cohort unless stated otherwise. Output notation is `output_index/output_count`; every listed row has count `1` unless specified.

| Evidence | Result |
| --- | --- |
| Source `10` has output `0/1` on each of two replicas | Pass independently on both replicas; do not sum counts |
| Group issues `[10, 11]`; source `10` has count `2`, source `11` is absent | Two issues, despite matching group totals |
| Source `10` has trained `0/3`, `2/3` and skipped `1/3` | Pass: every sibling has exactly one outcome |
| Source `10` has only `0/3`, `2/3` | Fail: missing sibling `1/3` |
| Source `10` has one drop and no consumption | Pass; two drops fail, and one drop plus any skipped/trained row also fails |
| Young or unissued source `10` is already observed with count `2` | Fail immediately; consumption bypasses grace |
| Young source `10` has no consumption | Wait for grace; absence becomes an error exactly at the boundary |
| Source `10` passed, then its next current snapshot shows count `2` | Fail on recheck; previous success is not cached |

## State and cost

- **Scope**: No custom replay allowance; intentionally consuming the same identity twice fails. Callers supply a coherent run's evidence; this rule does not partition by model identity or restore checkpoints.
- **Startup grace**: The analyzer wrapper can temporarily filter issuance predating process startup, separately from per-source maturity.
- **Storage**: Every call rebuilds state. Retaining issuance/drop history and bounding witness files are responsibilities outside this directory.
- **Time**: Scan input events and their issuance/drop entries, sort identities and replicas, group selected witness rows, then visit every eligible source on every replica. The final loop has `sources × replicas` cost in addition to row validation.
- **Memory**: Dictionaries scale with unique issuance/drop indices and selected witness rows; diagnostics can scale with `sources × replicas`. There is no persistent “already checked” cache.
