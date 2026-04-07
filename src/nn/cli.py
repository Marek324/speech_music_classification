# nn/cli.py
# CLI group for all neural network models.

import click

from .tcn.cli import tcn_group
from .own.cli import own_group


@click.group("nn")
def nn_group():
    """Neural network models (TCN, Own)."""
    pass


nn_group.add_command(tcn_group)
nn_group.add_command(own_group)
