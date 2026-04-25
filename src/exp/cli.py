# exp/cli.py
# Top-level CLI group for experiments.

import click

from .latency.cli import latency_group
from .nn_architecture.cli import nn_architecture_group
from .nn_preprocessor.cli import nn_preprocessor_group
from .small_tcn_pareto.cli import small_tcn_pareto_group
from .tcn_ablation.cli import tcn_ablation_group
from .tcn_combined.cli import tcn_combined_group
from .tcn_frontend.cli import tcn_frontend_group
from .tcn_hybrid.cli import tcn_hybrid_group
from .transitions.cli import transitions_group


@click.group("exp")
def exp_group():
    """Experiment submodules (ablations, variants)."""
    pass


exp_group.add_command(latency_group)
exp_group.add_command(nn_architecture_group)
exp_group.add_command(nn_preprocessor_group)
exp_group.add_command(small_tcn_pareto_group)
exp_group.add_command(tcn_ablation_group)
exp_group.add_command(tcn_combined_group)
exp_group.add_command(tcn_frontend_group)
exp_group.add_command(tcn_hybrid_group)
exp_group.add_command(transitions_group)
