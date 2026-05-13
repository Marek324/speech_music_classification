# nn/variants.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

from pathlib import Path
from typing import Any, Dict

import tomli

from .variant import Variant, make_variant

_HERE = Path(__file__).resolve().parent
_VARIANTS_TOML = _HERE / "variants.toml"
_ROOT_CONFIG = _HERE.parent.parent / "config.toml"


def _load_toml(path: Path) -> Dict[str, Any]:
    """Parse a TOML file, returning an empty dict if the file is missing."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomli.load(f)


_RAW = _load_toml(_VARIANTS_TOML)
_ROOT_DATASET = _load_toml(_ROOT_CONFIG).get("dataset", {})

VARIANTS: Dict[str, Variant] = {}
_exported_names: list[str] = ["VARIANTS"]

for _name, _spec in _RAW.get("variants", {}).items():
    _dataset = {**_ROOT_DATASET, **_spec.get("dataset", {})}
    _config = {"tcn": _spec.get("tcn", {}), "dataset": _dataset}

    _v = make_variant(
        name=_name,
        display_name=_spec["display_name"],
        ui_label=_spec.get("ui_label"),
        cli_group=_spec["cli_group"],
        group_help=_spec["group_help"],
        weights_stem=_spec["weights_stem"],
        subdir=_spec["subdir"],
        config=_config,
    )
    VARIANTS[_name] = _v

    _g = globals()
    _g[_spec["display_name"]]       = _v.cls
    _g[f"{_name}_group"]            = _v.group
    _g[f"get_{_name}_config"]       = _v.get_config
    _g[f"get_{_name}_weights_path"] = _v.get_weights_path
    _g[f"get_{_name}_stats_path"]   = _v.get_stats_path
    _g[f"_{_name.upper()}_DEFAULT"] = _v.DEFAULT

    _exported_names.extend([
        _spec["display_name"],
        f"{_name}_group",
        f"get_{_name}_config",
        f"get_{_name}_weights_path",
        f"get_{_name}_stats_path",
        f"_{_name.upper()}_DEFAULT",
    ])

__all__ = _exported_names
