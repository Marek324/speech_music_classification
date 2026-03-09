# main.py
# Marek Hric

import logging

import click

from .classic.cli import classic_group
from .logging_config import setup_logging
from .tcn.cli import tcn_group


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Speech/music classifier. Usage: smclassifier <approach> <model> <command>"""
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)


cli.add_command(classic_group)
cli.add_command(tcn_group)
