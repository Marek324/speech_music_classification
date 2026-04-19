# nn/cli.py
# CLI group for all neural network models.

import click

from .tcn.cli import tcn_group
from .tcn_lstm.cli import tcn_lstm_group
from .small_tcn.cli import small_tcn_group


@click.group("nn")
def nn_group():
    """Neural network models (TCN, TCNLSTM, SmallTCN)."""
    pass


nn_group.add_command(tcn_group)
nn_group.add_command(tcn_lstm_group)
nn_group.add_command(small_tcn_group)
