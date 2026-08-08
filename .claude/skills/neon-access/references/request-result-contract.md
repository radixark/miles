# Request and Result Contract

## Request

The local client compresses the exact UTF-8 SQL with deterministic gzip and sends these `workflow_dispatch` inputs:

- `request_id`: a canonical UUID identifying one request and its artifact.
- `sql_gzip_base64`: base64 of the gzip-compressed SQL bytes.
- `reason`: caller-supplied audit context; it is metadata and never becomes SQL.
- `recipient_cert_base64`: a one-use self-signed certificate whose private key remains in the local temporary directory.
- `max_result_bytes`: the maximum combined serialized size of returned row values.

The complete dispatch body must fit GitHub's 65,535-character input payload limit. This is a transport limit only and does not inspect SQL content.

## Result

The decrypted JSON document has `schema_version: 1` and includes `request_id`, `run_id`, `actor`, `reason`, `sql_sha256`, `status`, `results`, `truncated`, and `error`.

Each entry in `results` represents one PostgreSQL result set and contains `command_status`, `columns`, and `rows`. Statements without rows have empty `columns` and `rows` while retaining their command status.

JSON-native values remain native. Bytes use `{"type":"bytes","base64":"..."}`. Decimal, UUID, date/time, interval, and other non-native values use `{"type":"<python type>","value":"<string>"}`. This is result serialization, not SQL filtering.

When `max_result_bytes` is reached, row capture stops and `truncated` becomes `true`; SQL execution itself is not cancelled or changed.

## Encryption and Cleanup

The workflow encrypts the result with CMS and AES-256 before upload. The artifact is named `neon-access-<request_id>` and has one-day retention.

The local client decrypts the artifact, verifies the request ID, run ID, actor, and SQL SHA-256, then deletes that exact artifact. A cleanup warning does not invalidate a verified result, but the artifact remains downloadable in encrypted form until deletion or retention expiry.

The result's `status` is `error` when PostgreSQL returns an error. The encrypted payload contains the database error type, SQLSTATE when available, and message. The local client prints the payload and exits nonzero.
