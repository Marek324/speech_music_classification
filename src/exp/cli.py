# exp/cli.py
# Top-level CLI group for experiments.

import click

from .tcn_ablation.cli import tcn_ablation_group


@click.group("exp")
def exp_group():
    """Experiment submodules (ablations, variants)."""
    pass


exp_group.add_command(tcn_ablation_group)
