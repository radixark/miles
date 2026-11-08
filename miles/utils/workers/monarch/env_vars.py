"""Variables only a ``--cluster-backend monarch`` run reads.

This module imports nothing from torchmonarch, so the launcher and the argument checks, which run without it, can
read them too.
"""

# The names of the variables the launcher exports to a --cluster-backend monarch driver, comma separated; the driver
# passes them on to every worker process it spawns, since one on another host does not inherit the driver's
# environment, as ray's runtime env would pass them on.
MONARCH_RUN_ENV_VARS_ENV_VAR = "MILES_MONARCH_RUN_ENV_VARS"
# The saved state of the Monarch job a --cluster-backend monarch run runs in, which a launcher such as
# ``python -m miles.utils.workers.monarch.slurm`` submitted; the job started a worker loop on every one of its hosts.
MONARCH_JOB_STATE_ENV_VAR = "MILES_MONARCH_JOB_STATE"
