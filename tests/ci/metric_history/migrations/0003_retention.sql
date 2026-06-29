-- Retention for the CI metric-history tables.
--
-- Baselines only ever read the most recent N trusted runs per identity, so
-- rows older than the retention window carry no query value and only grow the
-- tables. This deletes runs (and their metric_values) outside a created_at
-- window.
--
-- Run as a privileged maintenance role on a schedule (e.g. a daily Neon cron /
-- scheduled job). The least-privilege app role from 0002 intentionally lacks
-- DELETE, so retention cannot run as the gate.
--
-- The window (90 days here) is a starting point; tune it to comfortably exceed
-- the largest baseline ``limit`` the gate uses multiplied by the run cadence,
-- so a full trusted baseline always remains available.

-- Child rows first to respect the metric_values -> runs foreign key.
DELETE FROM metric_values
WHERE run_id IN (
    SELECT run_id FROM runs
    WHERE created_at < now() - INTERVAL '90 days'
);

DELETE FROM runs
WHERE created_at < now() - INTERVAL '90 days';

-- Alternative for high-volume deployments: range-partition `runs` by
-- created_at (monthly) and DROP whole partitions instead of row-by-row
-- DELETE. That keeps retention O(1) per period and avoids vacuum churn, at the
-- cost of a partitioned-table migration. Not adopted yet -- the volume here
-- (a handful of runs per test per day) does not justify it.
