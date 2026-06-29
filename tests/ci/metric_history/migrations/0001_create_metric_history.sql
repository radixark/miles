-- Metric-history schema for the CI regression gate (PostgreSQL / Neon).
--
-- Two normalized tables. Apply this file once at provisioning time; the
-- application never issues DDL on the read/write path.

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    test_path           TEXT NOT NULL,
    backend             TEXT NOT NULL,
    suite               TEXT NOT NULL,
    test_file_hash      TEXT NOT NULL,            -- sha256 of the test file contents, computed by the caller
    commit_sha          TEXT NOT NULL,
    pr_number           INTEGER,
    github_run_id       BIGINT,
    github_run_attempt  INTEGER,
    event_name          TEXT,
    ref                 TEXT,
    created_at          TIMESTAMPTZ NOT NULL,
    trusted             BOOLEAN NOT NULL          -- run-level: a run is trusted as a whole or not at all
);

CREATE TABLE IF NOT EXISTS metric_values (
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    metric_key  TEXT NOT NULL,
    sub_label   TEXT,                             -- NULL = the single unlabeled measurement for this key
    value       DOUBLE PRECISION NOT NULL
);

-- Composite index serving the baseline query: it filters on the identity tuple
-- plus trusted and reads rows in created_at DESC order without a sort.
CREATE INDEX IF NOT EXISTS runs_baseline_idx
    ON runs (test_path, backend, suite, test_file_hash, trusted, created_at DESC);

-- Lookup index for joining metric_values back to its run.
CREATE INDEX IF NOT EXISTS metric_values_run_id_idx
    ON metric_values (run_id);
