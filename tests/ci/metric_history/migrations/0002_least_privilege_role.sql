-- Least-privilege application role for the CI metric-history gate.
--
-- The gate process only inserts runs, reads baselines, and flips the trusted
-- flag. It must never alter the schema, so this role is granted INSERT /
-- SELECT / UPDATE on the two tables and nothing more -- no CREATE / ALTER /
-- DROP, no DELETE. A stray runtime DDL or an accidental row delete fails as a
-- permission error instead of mutating data.
--
-- Run this as a privileged role at provisioning time, AFTER
-- 0001_create_metric_history.sql. Replace the password out-of-band; the DSN is
-- handed to the gate via the NEON_DATABASE_URL environment variable (a CI
-- secret), not stored here.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ci_metric_history_app') THEN
        CREATE ROLE ci_metric_history_app LOGIN PASSWORD 'CHANGE_ME_OUT_OF_BAND';
    END IF;
END
$$;

-- Connect + use the schema, but not create objects in it.
GRANT CONNECT ON DATABASE CURRENT_CATALOG TO ci_metric_history_app;
GRANT USAGE ON SCHEMA public TO ci_metric_history_app;

-- Exactly the verbs the gate uses. No DELETE, no TRUNCATE, no DDL.
GRANT INSERT, SELECT, UPDATE ON runs TO ci_metric_history_app;
GRANT INSERT, SELECT, UPDATE ON metric_values TO ci_metric_history_app;

-- Retention (0003) runs as a privileged maintenance role, not as this app
-- role, so the app role is deliberately NOT granted DELETE here.
