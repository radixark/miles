-- doc-dev: docs/ci/04-runtime-est-time.md
CREATE TABLE IF NOT EXISTS ci_test_runtime_attempts (
    test_path TEXT NOT NULL,
    backend TEXT NOT NULL,
    suite TEXT NOT NULL,
    test_attempt INTEGER NOT NULL CHECK (test_attempt >= 1),
    status TEXT NOT NULL CHECK (status IN ('PASS', 'FAIL', 'TIMEOUT')),
    elapsed_seconds DOUBLE PRECISION NOT NULL CHECK (elapsed_seconds >= 0),
    estimated_seconds DOUBLE PRECISION NOT NULL CHECK (estimated_seconds > 0),
    commit_sha TEXT NOT NULL,
    github_run_id BIGINT NOT NULL,
    github_run_attempt INTEGER NOT NULL CHECK (github_run_attempt >= 1),
    event_name TEXT NOT NULL,
    git_ref TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS ci_test_runtime_attempts_run_test_attempt_key
ON ci_test_runtime_attempts (
    github_run_id,
    github_run_attempt,
    test_path,
    backend,
    suite,
    test_attempt
);

CREATE INDEX IF NOT EXISTS ci_test_runtime_attempts_estimation_idx
ON ci_test_runtime_attempts (test_path, backend, suite, recorded_at DESC)
WHERE status = 'PASS'
  AND event_name = 'schedule'
  AND git_ref = 'refs/heads/main';
