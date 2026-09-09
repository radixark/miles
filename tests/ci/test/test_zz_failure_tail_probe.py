"""TEMPORARY: proves the suite summary carries a real failure. Not for merge.

Prints far more than the notifier's tail window, then fails, so a real CI run
exercises the whole chain: child pipe -> failure tail -> summary -> job log ->
notifier evidence.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

if __name__ == "__main__":
    for index in range(60_000):
        print(f"probe filler line {index} " + "x" * 60)
    raise AssertionError("failure-tail probe: the reactor exploded at step 42")
