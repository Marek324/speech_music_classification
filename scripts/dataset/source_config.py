# scripts/dataset/source_config.py
# Marek Hric

import tomllib
from dataclasses import dataclass, field, replace
from enum import Enum  # TODO: migrate project to 3.12 later to use StrEnum
from pathlib import Path
from typing import Any, Dict, Literal, Optional


class TierName(Enum):
    mid = "mid"
    full = "full"
    # `crit` is a hand-curated test-only tier (see scripts/dataset/crit/). Kept
    # in the enum so the staging-layout helpers and tier-keyed dicts work, but
    # it's hidden from the build.py CLI tier choices — crit is opted in/out
    # via --no-critical-set / --only-critical-set instead.
    crit = "crit"


# Parent directory for Hub upload: contains ``mid/``, ``full/`` (one subfolder per build).
HF_DATASET_STAGING_ROOT = Path("speech_music_dataset")


def tier_dataset_dir(staging_root: Path, tier: TierName) -> Path:
    """LibriSpeech-style: one folder per config (tier) under a shared staging root."""
    return staging_root / tier.value


RAND_SEED = 381


def seed_all() -> None:
    """Fix RNGs for ``datasets.shuffle``, numpy, torch (incl. CUDA if present)."""
    import random

    import numpy as np
    import torch

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RAND_SEED)


# --smoke: at most this many HF rows read per source (1 per split: train/validation/test)
SMOKE_SOURCE_MAX_ROWS = 3
# --smoke: each augmented recipe targets this many minutes
SMOKE_AUG_TARGET_MINUTES = 0.1

# ── multi-speaker augmentation (LibriMix-style, see augmentation.py) ──

MULTISPEAKER_GAIN_RANGE_DB: tuple[float, float] = (-5.0, 5.0)


@dataclass(frozen=True)
class MultispeakerAugEntry:
    """Synthetic multi-speaker recipe: mix pairs of single-speaker clips."""

    speech_subclass: str
    target_minutes: float
    output_subclass: str = "speech_multispeaker"


MULTISPEAKER_AUG_SOURCES: Dict[TierName, tuple[MultispeakerAugEntry, ...]] = {
    TierName.mid: (MultispeakerAugEntry("speech_clean", 36.0),),
    TierName.full: (MultispeakerAugEntry("speech_clean", 180.0),),
    TierName.crit: (),
}


def multispeaker_aug_entries(tier: TierName, *, smoke: bool = False) -> list[MultispeakerAugEntry]:
    """Copy of tier multispeaker recipes; ``smoke=True`` uses ``SMOKE_AUG_TARGET_MINUTES``."""
    entries = list(MULTISPEAKER_AUG_SOURCES[tier])
    if smoke:
        return [replace(e, target_minutes=SMOKE_AUG_TARGET_MINUTES) for e in entries]
    return entries


# ── speech-over-music augmentation (see augmentation.py) ──

MUSIC_RELATIVE_DB = -10.0
MUSIC_POOL_SIZE = 200  # FMA clips to pre-fetch (evenly across genres)

FMA_HF_ID = "rpmon/fma-genre-classification"
FMA_GENRES: tuple[str, ...] = (
    "Electronic",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "Pop",
    "Rock",
)

AUGMENT_MUSIC_GENRES: dict[TierName, tuple[str, ...]] = {
    TierName.mid: FMA_GENRES,
    TierName.full: FMA_GENRES,
}
FMA_GENRE_MAP: dict[str, int] = {
    "Electronic": 0,
    "Folk": 2,
    "Hip-Hop": 3,
    "Instrumental": 4,
    "Pop": 6,
    "Rock": 7,
}

# Splits on disk and in Hub ``load_dataset(..., split=...)``.
HF_SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")

# Per-tier train/val/test fractions (must sum to 1.0).
TIER_SPLIT_FRACTIONS: Dict[TierName, Dict[str, float]] = {
    TierName.mid:  {"train": 0.80, "validation": 0.10, "test": 0.10},
    TierName.full: {"train": 0.84, "validation": 0.08, "test": 0.08},
    # crit is test-only; the value is unused (crit_process hardcodes the test
    # writer) but the entry is required so TIER_SPLIT_FRACTIONS[tier_key]
    # lookups in build.py don't KeyError.
    TierName.crit: {"test": 1.0},
}


@dataclass(frozen=True)
class AugmentEntry:
    """Synthetic speech-over-music recipe: same idea as ``SourceEntry.target_minutes``."""

    output_subclass: str  # written ``subclass`` / folder semantics
    speech_subclass: str  # which base ``speech/{split}/speech`` rows to mix
    target_minutes: float


# Per-tier targets (minutes of *synthetic* audio per recipe), same pattern as ``sources.SOURCES``.
AUGMENT_SOURCES: Dict[TierName, tuple[AugmentEntry, ...]] = {
    TierName.mid: (
        AugmentEntry("speech_som", "speech_clean", 36.0),
        AugmentEntry("speech_msom", "speech_multispeaker", 24.0),
    ),
    TierName.full: (
        AugmentEntry("speech_som", "speech_clean", 180.0),
        AugmentEntry("speech_msom", "speech_multispeaker", 120.0),
    ),
    TierName.crit: (),
}


def augment_entries(tier: TierName, *, smoke: bool = False) -> list[AugmentEntry]:
    """Copy of tier recipes; ``smoke=True`` uses ``SMOKE_AUG_TARGET_MINUTES`` per recipe."""
    entries = list(AUGMENT_SOURCES[tier])
    if smoke:
        return [replace(e, target_minutes=SMOKE_AUG_TARGET_MINUTES) for e in entries]
    return entries


# ── speech-over-noise augmentation ──

# Speech-to-noise ratio range in dB (speech louder than noise).
# 5 dB = audibly noisy, 20 dB = lightly noisy.
NOISE_SNR_RANGE_DB: tuple[float, float] = (5.0, 20.0)


@dataclass(frozen=True)
class NoiseAugEntry:
    """Synthetic noisy-speech recipe: mix ``speech_subclass`` clips with background noise."""

    speech_subclass: str
    target_minutes: float
    output_subclass: str = "speech_noisy"


NOISE_AUG_SOURCES: Dict[TierName, tuple[NoiseAugEntry, ...]] = {
    TierName.mid: (
        NoiseAugEntry("speech_clean", 96.0),
    ),
    TierName.full: (
        NoiseAugEntry("speech_clean", 480.0),
    ),
    TierName.crit: (),
}


def noise_aug_entries(tier: TierName, *, smoke: bool = False) -> list[NoiseAugEntry]:
    """Copy of tier noise recipes; ``smoke=True`` uses ``SMOKE_AUG_TARGET_MINUTES``."""
    entries = list(NOISE_AUG_SOURCES[tier])
    if smoke:
        return [replace(e, target_minutes=SMOKE_AUG_TARGET_MINUTES) for e in entries]
    return entries


@dataclass
class SourceEntry:
    name: str
    hf_id: str
    cls: Literal["speech", "music", "background", "inactive"]
    subclass: str
    target_minutes: float  # minutes of audio to collect from this source
    detector: Literal["vad", "music", "silence"]
    split: str
    audio_decode: bool
    audio_col: str = "audio"
    filter_col: Optional[str] = None
    filter_val: Optional[Any] = None
    filter_include: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        slug = self.hf_id.replace("/", "_") if self.hf_id else "unset_source"
        sub = self.subclass or "default"
        return f"{slug}_{self.cls}_{sub}"

    @property
    def metadata_subclass(self) -> str:
        if self.cls == "speech":
            return f"speech_{self.subclass}" if self.subclass else "speech"
        if self.cls == "music":
            return f"music_{self.subclass}" if self.subclass else "music"
        return self.subclass or "noise"

    def max_rows_for_stream(self) -> int:
        """Estimate how many HF rows to stream to reach ``target_minutes``."""
        if self.detector == "vad":
            avg_s = 8.0
        elif self.detector == "music":
            avg_s = 25.0
        else:
            # Silence/noise clips (e.g. DEMAND) are often several minutes long;
            # use a conservative 60 s estimate so we don't over-fetch.
            avg_s = 60.0
        target_s = self.target_minutes * 60.0
        n = int(target_s / avg_s * 2) + 1
        if target_s < 90:
            return max(3, n)
        # Silence sources have large per-row payloads; a lower floor avoids
        # buffering gigabytes just to reach the target.
        if self.detector == "silence":
            return max(10, n)
        return max(100, n)

    def hf_load_kwargs(self) -> Dict[str, Any]:
        return dict(self.extra_kwargs)

    def cls_label(self) -> Literal[-1, 1, 2]:
        match self.cls:
            case "speech":
                return -1
            case "music":
                return 1
            case "background" | "inactive":
                 return 2

@dataclass
class TierConfig:
    entries: list[SourceEntry]


def load_sources(path: Path) -> Dict[TierName, TierConfig]:
    """Load source definitions from a TOML file structured as [tier.name] tables.

    Each ``[tier.name]`` block maps directly to a ``SourceEntry``; the source
    name is taken from the TOML key so it need not be repeated as a field.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    result: Dict[TierName, TierConfig] = {}
    for tier in TierName:
        entries = [
            SourceEntry(
                name=name,
                hf_id=fields["hf_id"],
                cls=fields["cls"],
                subclass=fields["subclass"],
                target_minutes=fields["target_minutes"],
                detector=fields["detector"],
                split=fields["split"],
                audio_decode=fields["audio_decode"],
                audio_col=fields.get("audio_col", "audio"),
                filter_col=fields.get("filter_col"),
                filter_val=fields.get("filter_val"),
                filter_include=fields.get("filter_include", True),
                extra_kwargs=fields.get("extra_kwargs", {}),
            )
            for name, fields in data.get(tier.value, {}).items()
        ]
        result[tier] = TierConfig(entries=entries)
    return result


_SOURCES_TOML = Path(__file__).parent / "sources.toml"
SOURCES: Dict[TierName, TierConfig] = load_sources(_SOURCES_TOML)
TIER_TOTAL_MINUTES: Dict[TierName, float] = {
    t: float(sum(e.target_minutes for e in SOURCES[t].entries)) for t in TierName
}


def tier_total_nominal_minutes(tier: TierName) -> float:
    """``SOURCES`` (HF) plus all synthetic augmentation targets — full dataset size."""
    ms = sum(e.target_minutes for e in MULTISPEAKER_AUG_SOURCES[tier])
    aug = sum(e.target_minutes for e in AUGMENT_SOURCES[tier])
    nz = sum(e.target_minutes for e in NOISE_AUG_SOURCES[tier])
    return TIER_TOTAL_MINUTES[tier] + ms + aug + nz


TIER_TOTAL_NOMINAL_MINUTES: Dict[TierName, float] = {
    t: tier_total_nominal_minutes(t) for t in TierName
}


def make_entries(tier: TierName) -> list[SourceEntry]:
    """Return the source list for a tier, raising if its target_minutes are zero."""
    entries = SOURCES[tier].entries
    if not entries or sum(e.target_minutes for e in entries) <= 0:
        raise ValueError(f"Tier {tier!r} has zero total target_minutes")
    return list(entries)

