---
name: neon-access
description: Run arbitrary PostgreSQL SQL against the Miles Neon database through the repository-authorized GitHub workflow and return the encrypted result. Use when a repository writer asks to query, inspect, update, migrate, or otherwise operate on Neon without exposing NEON_DATABASE_URL. The SQL is not classified, rewritten, split, or restricted; the database role remains the authorization boundary.
---

# Neon Access

Use the repository workflow as the only route to Neon. Never ask the user for `NEON_DATABASE_URL`, print it, read it locally, or pass it to the caller script.

## Run SQL

1. Confirm the exact SQL and a short audit reason with the user before dispatching. Treat every statement as potentially state-changing because this skill intentionally does not inspect or classify SQL.
2. Save the exact UTF-8 SQL in a temporary `.sql` file. Preserve the text byte-for-byte; do not normalize, split, wrap, retry, or add a transaction.
3. Confirm the active GitHub identity with `gh auth status`. The bundled client also checks that identity's live repository permission before dispatch.
4. From the repository root, run:

   ```bash
   python3 .claude/skills/neon-access/scripts/run_neon_workflow.py \
     --reason '<short audit reason>' \
     /absolute/path/to/request.sql
   ```

5. Return the decoded JSON result and the workflow URL. If the result status is `error`, report the database error and the possibility of partial effects; never automatically retry.

## Boundaries

- The caller and workflow both require legacy GitHub permission `write` or `admin`; GitHub maps the Maintain role to legacy `write`.
- The workflow reads the existing `secrets.NEON_DATABASE_URL`. Users do not retrieve or provide this secret.
- Any PostgreSQL SQL accepted by the server is allowed, including multiple statements, DDL, DML, functions, and explicit transaction control.
- `psql` backslash commands are client commands, not PostgreSQL SQL, and are unsupported.
- Execution uses autocommit and performs no implicit transaction wrapping. A later error can leave earlier statements committed.
- Limits apply only to transport size, runtime, and returned row bytes. They do not inspect or restrict SQL semantics.
- Results and database errors are encrypted for the one-time local key before artifact upload. Public logs and summaries contain metadata only.

Read [references/request-result-contract.md](references/request-result-contract.md) when debugging transport, result validation, or cleanup.
