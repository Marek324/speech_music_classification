# nn/own/cli.py
# CLI placeholder for custom NN model.

import click


@click.group("own")
def own_group():
    """Custom NN model (not yet implemented)."""
    pass


@own_group.command("train")
def train_cmd():
    """Train the own model (not implemented)."""
    raise NotImplementedError("OwnModel training is not yet implemented.")


@own_group.command("eval")
def eval_cmd():
    """Evaluate the own model (not implemented)."""
    raise NotImplementedError("OwnModel evaluation is not yet implemented.")


@own_group.command("smoke-test")
def smoke_test_cmd():
    """Smoke test the own model (not implemented)."""
    raise NotImplementedError("OwnModel smoke-test is not yet implemented.")
