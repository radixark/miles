# GLM-4.7 reasoning launchers

Experimental synchronous and fully asynchronous launchers for training the full
GLM-4.7 model on GSM8K. These scripts use the math reward path and do not depend
on SWE-Agent, Harbor, or an agent server.

- `run-glm47-reasoning.py`: colocated synchronous training.
- `run-glm47-reasoning-async.py`: disaggregated fully asynchronous training.
