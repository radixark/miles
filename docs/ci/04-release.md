---
title: Release a Version
description: Cut a versioned Miles release branch, run release CI, tag an exact release, and publish the official Docker images.
---

This runbook is for maintainers publishing an official Miles version from `radixark/miles`. It covers the supported path from a base-version bump through the two published Docker tags. The version vocabulary and pin ownership live in [Versions and Images](/developer/versions); CI selection details live in [Stage](/ci/00-stage) and [Labels](/ci/01-label).

## Before you start

- Authenticate `gh` to `radixark/miles` with permission to run workflows, push release refs, and open pull requests.
- Choose a base version `X.Y.Z` and an exact release version. The exact version may be `X.Y.Z`, `X.Y.ZrcN`, or `X.Y.Z.postN`; the release branch and `setup.py` always use the base version.
- Keep the release branch unchanged from the start of release CI until the tag workflow successfully pushes the tag. The called suite jobs currently check out the branch name independently, so a moving branch can invalidate which SHA the final status represents; the tag workflow also does not prove that a supplied SHA is still the branch tip.
- Treat `force=true` on the tag workflow as an audited emergency bypass, never as the normal path.

The commands below use deliberately invalid placeholders. Replace them before dispatching anything:

```bash
BASE_VERSION=X.Y.Z
EXACT_VERSION=X.Y.Z
RELEASE_BRANCH="release/v${BASE_VERSION}"
```

For a release candidate, keep `BASE_VERSION=X.Y.Z` and set `EXACT_VERSION=X.Y.ZrcN`. A post release similarly uses `EXACT_VERSION=X.Y.Z.postN`.

## 1. Merge the base-version bump

Run the helper workflow, or open an equivalent pull request that changes the single `version="..."` assignment in `setup.py`:

```bash
gh workflow run bot-bump-miles-version.yml -f new_version="${BASE_VERSION}"
```

Pass only `X.Y.Z` here. `setup.py` and the release branch carry the base version; prerelease and post-release suffixes belong only in `EXACT_VERSION` at tag time.

The workflow opens a PR against `main`. A PR created with `GITHUB_TOKEN` does not trigger `pr-test.yml`; close and reopen that PR once to start normal CI, then merge it. Do not cut the release branch until `main` contains the base version.

## 2. Cut the release branch and run release CI

Before dispatching, follow the image-affecting-change guidance in [Docker build](/ci/02-docker-build) and confirm that the latest successful timestamped CUDA 13 development image contains every required image-layer change. If a required build has not run, dispatch it and wait for success:

```bash
gh workflow run docker-build.yml -f variant=cu13 -f image_tag=dev
```

`release-branch-cut.yml` selects the newest `dev-<YYYYMMDDHHMM>` tag and falls back to mutable `dev` if none exists. Stop rather than accepting that fallback.

Dispatch the branch-cut workflow:

```bash
gh workflow run release-branch-cut.yml -f branch_name="${RELEASE_BRANCH}"
```

Add `-f commit_sha=FULL_MAIN_SHA` to cut from a specific commit already on `main`; otherwise the workflow uses the checked-out `main` tip. On the first dispatch, it creates `release/vX.Y.Z`, records the SGLang and Megatron-LM commits in `release-lock.json`, retags the preflighted development image as `release-vX.Y.Z-ci`, and commits the lockfile on the release branch. Re-dispatching an existing branch preserves its lockfile and tests its current tip.

While this run is active, do not push or cherry-pick anything onto the release branch. The workflow runs full-scope CUDA, CPU, and ROCm jobs with `cadence=release`, then records a `release-ci` commit status. [Stage](/ci/00-stage) and [Labels](/ci/01-label) own the cadence details; ROCm is a smoke signal because its dependencies remain baked into the image.

After the run is green, resolve the branch tip and copy the first column as `RELEASE_SHA`:

```bash
git ls-remote https://github.com/radixark/miles.git "refs/heads/${RELEASE_BRANCH}"
RELEASE_SHA=FULL_RELEASE_SHA
gh api "repos/radixark/miles/commits/${RELEASE_SHA}/status" --jq '[.statuses[] | select(.context == "release-ci")][0].state'
```

The final command must print `success`. Keep using this immutable SHA for the tag step.

## 3. Apply a release hotfix, if needed

Fixes land on `main` first. Cherry-pick exactly one merged PR or one commit already reachable from `main`:

```bash
gh workflow run bot-cherry-pick.yml -f pr_number=PR_NUMBER -f target_branch="${RELEASE_BRANCH}"
# Or use exactly one commit:
gh workflow run bot-cherry-pick.yml -f commit_sha=FULL_MAIN_SHA -f target_branch="${RELEASE_BRANCH}"
```

The workflow rejects commits that touch frozen image-layer files because release CI would still run inside the previously retagged CI image; `requirements.txt` is allowed because CI installs it at runtime and the release build bakes it. If a required fix is rejected for touching an image layer, publish it through a new base-version bump and a fresh branch cut instead of forcing it onto the existing branch.

After every successful cherry-pick, repeat step 2, keep the branch frozen during that run, and replace `RELEASE_SHA` with the newly verified tip. An older green status never authorizes the new commit.

## 4. Tag the verified SHA

Re-read the remote branch tip immediately before dispatch and require it to equal the verified SHA, then dispatch the tag workflow with that immutable SHA:

```bash
REMOTE_RELEASE_SHA=$(git ls-remote https://github.com/radixark/miles.git "refs/heads/${RELEASE_BRANCH}" | cut -f1)
if [ "${REMOTE_RELEASE_SHA}" != "${RELEASE_SHA}" ]; then
  echo "release branch moved; rerun release CI on the new tip" >&2
  exit 1
fi
gh workflow run release-tag.yml -f version="${EXACT_VERSION}" -f ref="${RELEASE_SHA}"
```

The workflow checks the version grammar, verifies that `setup.py` contains the base version, requires `release-ci=success` on the checked-out SHA, creates annotated tag `v${EXACT_VERSION}`, and explicitly dispatches the Docker release. Passing an immutable SHA fixes the checkout target, but it does not prove that SHA is still the release branch tip. Keep the branch frozen until this workflow succeeds; the comparison above is an operator check because the workflow does not repeat it immediately before pushing the tag.

Do not use `force=true` to bypass missing or failed release CI. If the status is absent, the branch changed or the release run has not finished; rerun step 2 on the final branch tip.

## 5. Verify the published images

`release-docker.yml` builds from `v${EXACT_VERSION}` and publishes:

- `radixark/miles:v${EXACT_VERSION}` for CUDA 13 on `linux/amd64` and `linux/arm64`.
- `radixark/miles:v${EXACT_VERSION}-cu12` for CUDA 12.9 on `linux/amd64`.

Verify both manifests after the workflow succeeds:

```bash
docker buildx imagetools inspect "radixark/miles:v${EXACT_VERSION}"
docker buildx imagetools inspect "radixark/miles:v${EXACT_VERSION}-cu12"
```

The versioned release does not move `dev`, `dev-cu12`, `latest`, or `latest-cu12`.

If the automatic Docker dispatch must be retried, keep its two manual inputs paired to the same immutable tag:

```bash
gh workflow run release-docker.yml -f version="${EXACT_VERSION}" -f ref="v${EXACT_VERSION}"
```

The current manual interface does not cross-check `version` against `ref`; a mismatch could publish one ref's source under another official version tag.
