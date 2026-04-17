# exp/cli.py
# Top-level CLI group for experiments.

import click

from .nn_architecture.cli import nn_architecture_group
from .nn_preprocessor.cli import nn_preprocessor_group
from .tcn_ablation.cli import tcn_ablation_group
from .tcn_frontend.cli import tcn_frontend_group


@click.group("exp")
def exp_group():
    """Experiment submodules (ablations, variants)."""
    pass


exp_group.add_command(nn_architecture_group)
exp_group.add_command(nn_preprocessor_group)
exp_group.add_command(tcn_ablation_group)
exp_group.add_command(tcn_frontend_group)
