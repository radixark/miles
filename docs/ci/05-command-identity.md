---
title: Command Identity
description: Who may run each PR-comment CI command — the identity bindings, access groups, tier constraints, and token identities behind the comment gateway.
---

# Command Identity

This page owns the identity and authorization rules of the PR-comment command gateway. What each command does, its syntax, and its failure semantics live in [Labels](/ci/01-label); this page answers only "who may run it, and as whom does it execute".

## Identity binding

Every command arrives as an `issue_comment` event on a pull request. The gateway accepts only comments whose author is a human `User` — bot authors are rejected, which also stops the gateway's own `github-actions[bot]` replies from re-triggering it — and requires the event `sender` to match the comment author. Three identity facts are bound from the event: the author's stable numeric user ID, their login, and the `author_association` GitHub stamped on the comment at creation time. An association outside GitHub's documented enum fails closed. Grants never match on login text: the event login exists only to re-query the live repository permission, and that lookup verifies the returned identity's numeric ID against the comment author before trusting the permission value; the policy's `users[].login` is display-only.

## Access groups

`.github/workflows/policies/comment-command-access.json` is an exact, default-deny ACL: each command selects one access group, and a command entry cannot select a handler, token capability, workflow, API endpoint, or shell command. The policy holds three groups:

| Group | Commands | Admits |
|---|---|---|
| `add_label_access` | `/run-ci-<x>`, `/bypass-fastfail` | live `write`/`admin`, or an explicit numeric user `id` entry |
| `repo_write_access` | `/clear-labels` | live `write`/`admin` |
| `prior_contributor_access` | `/rerun-test <test-file>`, `/rerun-failed-ci` | comment `author_association` in `OWNER`/`MEMBER`/`COLLABORATOR`/`CONTRIBUTOR`, an explicit numeric user `id` entry, or live `write`/`admin` |

`add_label_access` and `prior_contributor_access` each contain an independent `users` allowlist whose entries have the shape `{"id": 123, "login": "alice"}`. Authorization uses only the stable numeric `id`; `login` is a required display annotation and may be stale without changing access. These entries let workflow owners grant a specific user one existing command tier without granting repository write: the label list grants only label commands, while the prior-contributor list grants only the two rerun commands. `repo_write_access` has no allowlist, so `/clear-labels` always requires live `write`/`admin`. GitHub reports the `maintain` role as legacy `write`; custom roles follow their base repository access.

This policy is checked in and reviewed as trusted configuration. Maintainers must keep each `users` entry to one unique positive numeric `id` plus a non-empty `login` annotation; the gateway deliberately does not verify that annotation against GitHub. Moving this policy to an external or dynamically updated source would require a separate design for validation, update authority, and auditability.

## Tier constraints

The tiers are deliberate, not incidental:

- **Label mutations stay write-gated because a label is CI policy.** A `run-ci-*` label selects what CI spends and, for fork PRs, doubles as the standing Approve-and-run decision (see [Labels](/ci/01-label)); granting it below write would let non-maintainers set policy.
- **The rerun commands are cheaper to grant, so they sit at the prior-contributor tier.** The author association GitHub stamps on the comment admits `OWNER`, `MEMBER`, `COLLABORATOR`, and `CONTRIBUTOR` — anyone with a commit already merged into `radixark/miles` — without a live-permission lookup; an explicit numeric user `id` entry admits a named exception, and anyone else needs live `write` or `admin`. First-time contributors and users with no history in the repository are denied unless they are explicitly allowlisted. A failed-job rerun re-executes an already-authorized run with its original privileges and SHA, and a file run is bounded by one registered file, its registration's timeout, and per-PR serialization, so the residual exposure is runner time rather than code trust.
- **Constraint: a fork head is the normal shape of a contribution from someone without write permission** — pushing a same-repository branch already requires write. `/rerun-test` therefore adds no fork-specific approval: the policy tier is the whole gate for both head shapes, and requiring anything extra of forks would exclude exactly the contributors the command exists for. What changes on a fork head is containment, not authorization: the head identity is re-verified against its unique open pull request before dispatch, and the run receives no repository secrets (`WANDB_API_KEY` and `HF_TOKEN` are withheld), matching the pr-test fork policy.

## Evaluation points

The handler evaluates the caller against the checked-in policy twice: once in the preflight job, and again in the capability-specific job immediately before the command mutates GitHub state. An explicit `users[].id` match uses the numeric comment-author identity bound to the event; `users[].login` is never compared with the event. An admitted author association is likewise the value bound to the comment; only the fallback path performs a live repository-permission lookup, and each lookup is a point-in-time result.

## Token identity

Authorization decides who may ask; token identity decides as whom the gateway acts. `/rerun-test` and `/rerun-failed-ci` execute on the workflow's own `GITHUB_TOKEN` with job-scoped permissions: the command job has `actions: write` plus `pull-requests: read`.

The reaction and file-run status jobs have `issues: write` plus `pull-requests: write`, because GitHub gates issues-API calls whose target issue is a pull request on the pull-requests scope. The final file-run status job adds `actions: read` to calculate elapsed time from its run. Their mutations and feedback therefore appear as `github-actions[bot]`.

Label commands execute on a command-App token minted with `Issues: write` plus `Pull requests: write`: a label added with `GITHUB_TOKEN` would never fire the `pull_request(labeled)` CI workflows, and the label lands on a pull request, whose issues-API mutations GitHub gates on the pull-requests scope. `Pull requests: write` also lets its holder submit reviews, including approvals. The handler never calls a review endpoint, the token exists only for the label-command job and is revoked when that job ends, and `main` requires one approving review from a code owner, which an App cannot be, so an App approval cannot satisfy the merge gate on its own. Neither token reaches the jobs that execute PR code, and the App private key never leaves the label path.

## Non-goals

The gateway neither widens nor narrows GitHub's own permission model: a user's existing UI/API label rights are untouched, and a denied command says nothing about what that user may do through GitHub directly.
