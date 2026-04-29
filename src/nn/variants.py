# nn/variants.py
# Variant registry. To add a new variant: append a [variants.<name>] block
# to variants.toml. No Python edits required — this module derives every
# per-variant symbol name from the TOML key + display_name.
#
# Dataset is sourced once from the repo-root config.toml[dataset] and merged
# into every variant's DEFAULT. A variant may override by adding a
# [variants.<name>.dataset] section, but the project default is not to repeat it.
#
# Exported per variant (for a TOML key `name` with `display_name = "Disp"`):
#
#     Disp                      — model class
#     <name>_group              — Click command group
#     get_<name>_config         — config loader (accepts overrides)
#     get_<name>_weights_path   — path helper
#     get_<name>_stats_path     — path helper
#     _<NAME>_DEFAULT           — frozen merged snapshot
#
# Plus `VARIANTS`, a dict[str, Variant] keyed by TOML variant name.

from pathlib import Path
from typing import Any, Dict

import tomli

from .variant import Variant, make_variant

_HERE = Path(__file__).resolve().parent
_VARIANTS_TOML = _HERE / "variants.toml"
_ROOT_CONFIG = _HERE.parent.parent / "config.toml"


def _load_toml(path: Path) -> Dict[str, Any]:
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
