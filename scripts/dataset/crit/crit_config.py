"""Critical-set manifest loader.

Mirrors the shape of source_config.SourceEntry / load_sources, but for
hand-curated local recordings rather than HuggingFace streams. The manifest is
read from ``crit/manifest.toml``; each TOML block becomes a CritEntry.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


_DETECTORS = ("vad", "music", "silence")
_CLASSES = ("speech", "music", "inactive")


@dataclass
class CritEntry:
    name: str
    file: Path
    cls: Literal["speech", "music", "inactive"]
    subclass: str
    detector: Optional[Literal["vad", "music", "silence"]] = None
    labels: Optional[list[dict[str, Any]]] = None
    source_url: Optional[str] = None
    license: Optional[str] = None
    notes: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def metadata_subclass(self) -> str:
        # Matches SourceEntry.metadata_subclass in source_config.py:198-204.
        if self.cls == "speech":
            return f"speech_{self.subclass}" if self.subclass else "speech"
        if self.cls == "music":
            return f"music_{self.subclass}" if self.subclass else "music"
        return f"noise_{self.subclass}" if self.subclass else "noise"


def _validate(entry: CritEntry) -> None:
    if entry.cls not in _CLASSES:
        raise ValueError(f"crit entry {entry.name!r}: cls must be one of {_CLASSES}, got {entry.cls!r}")
    if not entry.subclass:
        raise ValueError(f"crit entry {entry.name!r}: subclass is required")
    has_detector = entry.detector is not None
    has_labels = entry.labels is not None
    if has_detector == has_labels:
        raise ValueError(
            f"crit entry {entry.name!r}: set exactly one of `detector` or `labels` "
            f"(got detector={entry.detector!r}, labels={'set' if has_labels else 'unset'})"
        )
    if has_detector and entry.detector not in _DETECTORS:
        raise ValueError(f"crit entry {entry.name!r}: detector must be one of {_DETECTORS}")
    if has_labels:
        for i, lab in enumerate(entry.labels):
            if not isinstance(lab, dict):
                raise ValueError(f"crit entry {entry.name!r}: labels[{i}] must be a TOML table")
            for key in ("label", "start", "end"):
                if key not in lab:
                    raise ValueError(f"crit entry {entry.name!r}: labels[{i}] missing key {key!r}")
            if int(lab["start"]) > int(lab["end"]):
                raise ValueError(
                    f"crit entry {entry.name!r}: labels[{i}] has start > end "
                    f"({lab['start']} > {lab['end']})"
                )


def load_manifest(path: Path) -> list[CritEntry]:
    """Parse the manifest TOML; resolve `file` relative to the manifest's dir.

    Missing audio files are reported up front so the build doesn't fail mid-run
    on the Nth recording. Returns an empty list cleanly for an empty/comment-only
    manifest, which is the expected state of the scaffold before the user adds
    any recordings.
    """
    manifest_dir = path.parent
    if not path.exists():
        return []
    with open(path, "rb") as f:
        data = tomllib.load(f)

    entries: list[CritEntry] = []
    missing: list[str] = []
    for name, fields in data.items():
        if not isinstance(fields, dict):
            # Skip top-level scalars; manifest is meant to be all [section] blocks.
            continue
        file_path = (manifest_dir / fields["file"]).resolve() if "file" in fields else None
        if file_path is None or not file_path.exists():
            missing.append(f"{name} → {fields.get('file', '<no file field>')}")
            continue
        entry = CritEntry(
            name=name,
            file=file_path,
            cls=fields["cls"],
            subclass=fields["subclass"],
            detector=fields.get("detector"),
            labels=fields.get("labels"),
            source_url=fields.get("source_url"),
            license=fields.get("license"),
            notes=fields.get("notes"),
        )
        _validate(entry)
        entries.append(entry)

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            f"crit manifest {path}: {len(missing)} missing audio file(s):\n  {joined}"
        )
    return entries
