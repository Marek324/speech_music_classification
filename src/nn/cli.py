# nn/cli.py
# CLI group for all neural network models.

import click

from .tcn.cli import tcn_group
from .variants import VARIANTS


@click.group("nn")
def nn_group():
    """Neural network models (TCN + configured variants)."""
    pass


nn_group.add_command(tcn_group)
for _v in VARIANTS.values():
    nn_group.add_command(_v.group)
