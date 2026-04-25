"""Click entrypoint that launches the Reflex demo UI.

Single command: ``uv run smclassifier demo``. On first run (or whenever the
scaffolded ``.web/`` directory is missing) it transparently runs
``reflex init`` before starting the dev server.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import click

log = logging.getLogger(__name__)

_DEMO_DIR = Path(__file__).resolve().parent
_WEB_DIR = _DEMO_DIR / ".web"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(_DEMO_DIR.parent.parent), env.get("PYTHONPATH", "")])
    )
    return env


def _run_init() -> None:
    log.info("First-time setup: running `reflex init` in %s", _DEMO_DIR)
    cmd = [sys.executable, "-m", "reflex", "init", "--template", "blank"]
    result = subprocess.run(cmd, cwd=str(_DEMO_DIR), env=_env())
    if result.returncode != 0:
        raise click.ClickException(
            f"`reflex init` failed (exit code {result.returncode}). See output above."
        )


@click.command("demo")
@click.option("--port", default=3000, type=int, help="Frontend port.")
@click.option("--backend-port", default=8000, type=int, help="Backend port.")
@click.option("--prod", is_flag=True, default=False, help="Run a production build.")
def demo_cmd(port: int, backend_port: int, prod: bool):
    """Launch the streaming demo web UI (auto-inits on first run)."""
    if not _WEB_DIR.exists():
        _run_init()

    cmd = [
        sys.executable, "-m", "reflex", "run",
        "--frontend-port", str(port),
        "--backend-port", str(backend_port),
        "--env", "prod" if prod else "dev",
    ]
    log.info("Launching Reflex from %s: %s", _DEMO_DIR, " ".join(cmd))
    subprocess.run(cmd, cwd=str(_DEMO_DIR), env=_env(), check=False)
