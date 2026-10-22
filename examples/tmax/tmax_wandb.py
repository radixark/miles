"""The W&B run a TMax experiment logs to: its id, and its URL, which beaker/submit.py puts in the experiment's
description before the run exists. The launcher passes the same id, so both always agree."""

import re


def run_id(exp_name: str) -> str:
    """W&B run id for an experiment: every restart of a named experiment continues this one run."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", exp_name)[:64]


def run_url(*, entity: str, project: str, exp_name: str) -> str:
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id(exp_name)}"
