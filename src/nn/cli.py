# nn/cli.py
# CLI group for all neural network models.

import click

from .tcn.cli import tcn_group
from .own.cli import own_group
from .own2.cli import own2_group


@click.group("nn")
def nn_group():
    """Neural network models (TCN, Own, Own2)."""
    pass


nn_group.add_command(tcn_group)
nn_group.add_command(own_group)
nn_group.add_command(own2_group)
