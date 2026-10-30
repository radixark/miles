# Miles Code Maintenance Model

This document describes the code maintenance model for the Miles project. Miles couples
Megatron-LM for training with SGLang for rollout, so most changes touch both a trainer
backend and a rollout backend; this model is designed to keep reviews responsive across
those boundaries and to let maintainers shepherd PRs that cross area boundaries.

## Role Descriptions

There are four roles in this maintenance model. Some are custom roles, while others are
predefined by GitHub.

- **Merge Oncall**: drives the PR merge process for a specific area. Strong
  area-specific expertise; upholds a high bar for code quality.
  - Permission: Merge PRs. Bypass branch protection rules if needed.
  - Responsibility: Shepherd the merge of PRs assigned to their area. Revert or hotfix
    any issues related to their merge (especially if they bypass).
- **Codeowner**: protects critical code. Without a bypass, each PR needs at least one
  Codeowner approval for every modified file protected by [CODEOWNERS](./CODEOWNERS).
  This role is not an honor but a significant responsibility — PRs cannot be merged
  without your approval (except when bypassed by a Merge Oncall).
  - Permission: Approve PRs, allowing them to be merged without a bypass.
  - Responsibility: Review PRs in a timely manner.
- **Write**: has write permission to the Miles repo.
  - Permission: Merge PRs that have passed required tests and been approved by
    Codeowners. This role cannot bypass branch protection rules.
  - Responsibility: Review and merge PRs in a timely manner.
- **CI Oncall**: manages CI runners for specific hardware platforms.
  - Permission: Add CI runners.
  - Responsibility: Keep the CI runners up and running.

__Note__: Difference between Merge Oncall and Codeowner
- The Merge Oncall is an active role held by someone who actively tries to help merge
  PRs and can bypass CI if needed.
- The Codeowner is a passive protection role provided by GitHub; it prevents
  accidental changes to critical code.
- The list of Merge Oncalls is attached below. The list of Codeowners is in the
  [CODEOWNERS](./CODEOWNERS) file.

## Pull Request Merge Process

1. The author submits a pull request (PR) and fills out the PR checklist.
2. GitHub automatically requests reviews from Codeowners. A Merge Oncall picks up the
   PR (or is assigned by a maintainer) and shepherds it through review.
3. The PR runs CI; if a runner is offline or flaky, the author can ping the relevant
   CI Oncall.
4. The Merge Oncall coordinates the review (asking people to review) and approves the
   PR; the Codeowners also approve. If the assigned Merge Oncall is not responsive,
   the author can ping other related Merge Oncalls and Reviewers in the list below.
5. The code is merged:
   - **Ideal case**: every modified file has at least one Codeowner approval and CI
     is green. Anyone with write permission can merge.
   - **Exception**: when meeting all requirements is hard (flaky CI, slow reviews),
     a Merge Oncall can bypass branch protection to merge the PR — and is then on
     the hook for any follow-up revert or hotfix.

If you hit issues during the merge, ping `#miles-rl` on the [SGLang Slack](https://slack.sglang.ai).

## The List of Merge Oncalls and Reviewers

This section lists the oncalls for each module or feature. The format is
@github-username.

### Trainer (Megatron backend)
@fzyzcjy, @yueming-yuan,
@maocheng23,
@yushengsu-thu,
@Zhichenzzz

related files
- train.py
- miles/backends/megatron_utils/

### Rollout (SGLang backend)
@fzyzcjy, @yueming-yuan,
@maocheng23,
@yushengsu-thu,
@Zhichenzzz,
@guapisolo

related files
- miles/backends/sglang_utils/
- miles/rollout/

### Ray actors and orchestration
@fzyzcjy, @yueming-yuan,
@maocheng23

related files
- miles/ray/

### Router
@fzyzcjy, @yueming-yuan,
@guapisolo

related files
- miles/router/

### Multi-turn / rollout sessions
@fzyzcjy, @yueming-yuan,
@guapisolo,
@maocheng23,
@jybsuper

related files
- miles/rollout/session/

### Utils
@fzyzcjy, @yueming-yuan,
@guapisolo,
@maocheng23,
@jybsuper,
@Zhichenzzz

related files
- miles/utils/

### CI, Release, Package
@yushengsu-thu

related files
- .github/workflows/
- requirements.txt
- setup.py


### Other Notes

This list is based on the current situation. If you or someone you know would like to
take on more responsibility and are qualified, please ping
@Ying1123 and @fzyzcjy in
the Slack channel. They will start a nomination and internal review process.


## CI Maintenance Mode

When CI is unhealthy (for example, the scheduled tests on `main` are broken for
consecutive runs), the project enters **CI Maintenance Mode** by opening a pinned
tracking issue. While the mode is active:
- All PR CI runs are paused; resources are allocated to PRs that fix the CI.
- **Merging non-CI-fix PRs is prohibited.** Only PRs that fix the CI may be merged. In
  severe cases, merge permissions may be revoked.

Maintenance mode ends when CI is all green on `main` and the tracking issue is closed.

### Rebase-Required Mode

When a major update lands on `main` and all open PRs must rebase before CI can run
(without fully pausing CI), add a `MIN_BASE_SHA: <sha>` directive to the body of the
tracking issue. **The rebase check is enforced regardless of whether the issue is open
or closed.** While the directive is present:
- CI is allowed to run only for PRs whose branch already contains `<sha>` (GitHub
  compare API status `ahead` or `identical`).
- PRs that are `behind` or `diverged` are blocked with a "rebase required" error until
  they rebase onto the latest `main`.
- A `bypass-maintenance` label still bypasses this check for CI-fix PRs.

Notes:
- Only the **first** `MIN_BASE_SHA:` line in the issue body is read.
- The SHA must be 7-40 hex characters; malformed values are ignored.
- Avoid pasting the directive inside a fenced code block in the issue body — the
  parser does not skip code fences and may match example snippets.

Remove the directive from the issue body to lift the rebase requirement (closing the
issue alone does not lift it).
