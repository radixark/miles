"""Train-pass plan for Predictive Routing Replay (PR²).

The actor runs a single train pass over each rollout batch and selects the
predictive mode per mini-step via ``get_predictive_train_mode_for_step``:
mini-step ``step_id=0`` runs in ``SKIP_PREDICTIVE`` (matches paper Algorithm
3 line 3 condition ``i=1``), mini-steps ``step_id>=1`` run in
``COMPUTE_PREDICTIVE_LOSS``.
"""


def get_predictive_train_mode_for_step(*, role: str, predictive_enabled: bool, step_id: int) -> str:
    if role == "actor" and predictive_enabled and step_id == 0:
        return "skip"
    return "compute"
