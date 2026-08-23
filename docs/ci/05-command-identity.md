---
title: Command Identity
description: Who may run each PR-comment CI command — the identity bindings, access groups, tier constraints, and token identities behind the comment gateway.
---

# Command Identity

This page owns the identity and authorization rules of the PR-comment command gateway. What each command does, its syntax, and its failure semantics live in [Labels](/ci/01-label); this page answers only "who may run it, and as whom does it execute".

## Identity binding

Every command arrives as an `issue_comment` event on a pull request. The gateway accepts only comments whose author is a human `User` — bot authors are rejected, which also stops the gateway's own `github-actions[bot]` replies from re-triggering it — and requires the event `sender` to match the comment author. Three identity facts are bound from the event: the author's stable numeric user ID, their login, and the `author_association` GitHub stamped on the comment at creation time. An association outside GitHub's documented enum fails closed. Grants never match on login text: the login exists only to re-query the live repository permission, and that lookup verifies the returned identity's numeric ID against the comment author before trusting the permission value.

## Access groups

`.github/workflows/policies/comment-command-access.json` is an exact, default-deny ACL: each command selects one access group, and a command entry cannot select a handler, token capability, workflow, API endpoint, or shell command. The policy holds three groups:

| Group | Commands | Admits |
|---|---|---|
| `add_label_access` | `/run-ci-<x>`, `/bypass-fastfail` | live `write`/`admin`, or an explicit numeric `user_ids` entry |
| `repo_write_access` | `/clear-labels` | live `write`/`admin` |
| `prior_contributor_access` | `/rerun-test <test-file>`, `/rerun-failed-ci` | comment `author_association` in `OWNER`/`MEMBER`/`COLLABORATOR`/`CONTRIBUTOR`, else live `write`/`admin` |

Only `add_label_access` can contain explicit `user_ids`; those IDs let workflow owners grant a specific contributor label-command access without granting repository write, and they never grant any non-label operation. GitHub reports the `maintain` role as legacy `write`; custom roles follow their base repository access.

## Tier constraints

The tiers are deliberate, not incidental:

- **Label mutations stay write-gated because a label is CI policy.** A `run-ci-*` label selects what CI spends and, for fork PRs, doubles as the standing Approve-and-run decision (see [Labels](/ci/01-label)); granting it below write would let non-maintainers set policy.
- **The rerun commands are cheaper to grant, so they sit at the prior-contributor tier.** The author association GitHub stamps on the comment admits `OWNER`, `MEMBER`, `COLLABORATOR`, and `CONTRIBUTOR` — anyone with a commit already merged into `radixark/miles` — without a live-permission lookup; anyone else needs live `write` or `admin`. First-time contributors and users with no history in the repository are denied. A failed-job rerun re-executes an already-authorized run with its original privileges and SHA, and a file run is bounded by one registered file, its registration's timeout, and per-PR serialization, so the residual exposure is runner time rather than code trust.
- **Constraint: a fork head is the normal shape of a contribution from someone without write permission** — pushing a same-repository branch already requires write. `/rerun-test` therefore adds no fork-specific approval: the policy tier is the whole gate for both head shapes, and requiring anything extra of forks would exclude exactly the contributors the command exists for. What changes on a fork head is containment, not authorization: the head identity is re-verified against its unique open pull request before dispatch, and the run receives no repository secrets (`WANDB_API_KEY` and `HF_TOKEN` are withheld), matching the pr-test fork policy.

## Evaluation points

The handler evaluates the caller against the checked-in policy twice: once in the preflight job, and again in the capability-specific job immediately before the command mutates GitHub state. An explicit `user_ids` match uses the numeric comment-author identity bound to the event; an admitted author association is likewise the value bound to the comment; only the fallback path performs a live repository-permission lookup, and each lookup is a point-in-time result.

## Token identity

Authorization decides who may ask; token identity decides as whom the gateway acts. `/rerun-test` and `/rerun-failed-ci` execute on the workflow's own `GITHUB_TOKEN` with job-scoped permissions: the command job has `actions: write` plus `pull-requests: read`.

The reaction and file-run status jobs have `issues: write` plus `pull-requests: write`, because GitHub gates issues-API calls whose target issue is a pull request on the pull-requests scope. The final file-run status job adds `actions: read` to calculate elapsed time from its run. Their mutations and feedback therefore appear as `github-actions[bot]`.

Label commands execute on a command-App token minted with `Issues: write` only, because a label added with `GITHUB_TOKEN` would never fire the `pull_request(labeled)` CI workflows. Neither token reaches the jobs that execute PR code, and the App private key never leaves the label path.

## Non-goals

The gateway neither widens nor narrows GitHub's own permission model: a user's existing UI/API label rights are untouched, and a denied command says nothing about what that user may do through GitHub directly.
