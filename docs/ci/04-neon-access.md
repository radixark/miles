# Neon SQL Access

## Goal

The Neon access workflow gives Miles repository writers a deliberate path to execute arbitrary PostgreSQL SQL with the existing repository secret and receive the result without exposing the connection string or plaintext result through GitHub Actions.

This facility is intentionally not a query policy engine. It does not classify, parse, split, rewrite, wrap, or retry SQL. PostgreSQL and the role encoded by `NEON_DATABASE_URL` decide which operations are authorized.

## Authorization and Credentials

The supported entry point is `.claude/skills/neon-access/scripts/run_neon_workflow.py`. The client uses the caller's existing `gh` authentication, confirms the live repository permission is legacy `write` or `admin`, and dispatches `.github/workflows/neon-access.yml` on the default branch.

The workflow independently checks both `github.actor` and `github.triggering_actor` through the repository collaborator-permission API. This preserves the write gate on reruns, where the initial and triggering identities can differ.

The workflow reads `secrets.NEON_DATABASE_URL` only in the executor step. The local skill, user, logs, summaries, later steps, and artifacts do not receive that secret.

Workflow source and `.github/scripts/neon_access_job.py` are protected by the workflow owners in `.github/CODEOWNERS`. Repository branch protection remains responsible for requiring those reviews before changes reach the default branch.

## SQL Execution Semantics

The client encodes the exact UTF-8 file bytes with gzip and base64. The executor decodes them and calls `cursor.execute(sql_text, prepare=False)` once on a psycopg connection configured with `autocommit=True` and `prepare_threshold=None`.

PostgreSQL simple-query execution supports multiple statements and returns every result set exposed by psycopg. DDL, DML, queries, function calls, session statements, and explicit transaction control are passed through unchanged. `psql` backslash commands are not PostgreSQL SQL and are not supported.

There is no implicit transaction. If a multi-statement request fails after earlier work committed, the workflow reports the error but does not undo earlier effects. The client never retries a request automatically.

The only workflow limits are GitHub's dispatch payload capacity, the 15-minute job timeout, and the caller-selected returned-row byte limit. Reaching the row limit sets `truncated: true`; it does not cancel or change SQL execution.

## Result Delivery

For each invocation, the client creates a one-use RSA private key and certificate in a mode-`0700` temporary directory. Only the certificate is sent to GitHub.

The executor serializes every result set to JSON. Database errors are also represented inside this result, including the error type, SQLSTATE when available, and message. Workflow logs and the job summary contain request metadata only.

The workflow encrypts the JSON with CMS using AES-256, deletes the plaintext, and uploads `neon-access-<request_id>` with one-day retention. The client downloads and decrypts it, verifies the request ID, run ID, actor, and SQL SHA-256, then requests immediate artifact deletion.

GitHub artifact access is repository-wide rather than requester-only. Encryption is therefore the confidentiality boundary; immediate deletion and one-day retention only reduce the encrypted artifact's exposure window.

## Usage

Put the exact SQL in a local file and run:

```bash
python3 .claude/skills/neon-access/scripts/run_neon_workflow.py \
  --reason 'describe the intended operation' \
  /absolute/path/to/request.sql
```

Use `--max-result-bytes` to override the default 4 MiB returned-row limit. A nonzero exit means either the transport failed or the encrypted result has `status: error`; inspect the printed JSON before deciding whether any follow-up is safe.

Do not manually supply or retrieve `NEON_DATABASE_URL`. The workflow consumes the repository secret directly.

## Coupled Files

- `.github/workflows/neon-access.yml` owns dispatch authorization, secret scope, encryption, upload, and public metadata.
- `.github/scripts/neon_access_job.py` owns SQL decoding, exact execution semantics, result serialization, and the encrypted error payload source.
- `.claude/skills/neon-access/` owns the caller workflow, local key lifecycle, identity validation, artifact download, result verification, and cleanup.
- `tests/ci/test/test_neon_access.py` locks the arbitrary-SQL pass-through and encrypted delivery contracts without connecting to Neon.
