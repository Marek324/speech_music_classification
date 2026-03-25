from dataclasses import dataclass, field, replace
from enum import Enum  # TODO: migrate project to 3.12 later to use StrEnum
from pathlib import Path
from typing import Any, Dict, Literal, Optional


class TierName(Enum):
    mini = "mini"
    mid = "mid"
    full = "full"


# Parent directory for Hub upload: contains ``mini/``, ``mid/``, ``full/`` (one subfolder per build).
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


# --test: at most this many HF rows read per source
TEST_SOURCE_MAX_ROWS = 1
# --test: each augmented recipe targets this many minutes (like TEST for base sources)
TEST_AUG_TARGET_MINUTES = 0.1

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
    TierName.mini: {"train": 0.70, "validation": 0.15, "test": 0.15},
    TierName.mid:  {"train": 0.80, "validation": 0.10, "test": 0.10},
    TierName.full: {"train": 0.84, "validation": 0.08, "test": 0.08},
}


@dataclass(frozen=True)
class AugmentEntry:
    """Synthetic speech-over-music recipe: same idea as ``SourceEntry.target_minutes``."""

    output_subclass: str  # written ``subclass`` / folder semantics
    speech_subclass: str  # which base ``speech/{split}/speech`` rows to mix
    target_minutes: float


# Per-tier targets (minutes of *synthetic* audio per recipe), same pattern as ``sources.SOURCES``.
AUGMENT_SOURCES: Dict[TierName, tuple[AugmentEntry, ...]] = {
    TierName.mini: (
        AugmentEntry("speech_som", "speech_clean", 1.0),
    ),
    TierName.mid: (
        AugmentEntry("speech_som", "speech_clean", 36.0),
        AugmentEntry("speech_msom", "speech_multispeaker", 24.0),
    ),
    TierName.full: (
        AugmentEntry("speech_som", "speech_clean", 360.0),
        AugmentEntry("speech_msom", "speech_multispeaker", 240.0),
    ),
}


def augment_entries(tier: TierName, *, test: bool = False) -> list[AugmentEntry]:
    """Copy of tier recipes; ``test=True`` uses ``TEST_AUG_TARGET_MINUTES`` per recipe (smoke)."""
    entries = list(AUGMENT_SOURCES[tier])
    if test:
        return [replace(e, target_minutes=TEST_AUG_TARGET_MINUTES) for e in entries]
    return entries


@dataclass
class SourceEntry:
    name: str
    hf_id: str
    cls: Literal["speech", "music", "inactive"]
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
        if self.detector == "vad":
            avg_s = 8.0
        elif self.detector == "music":
            avg_s = 25.0
        else:
            avg_s = 15.0
        target_s = self.target_minutes * 60.0
        n = int(target_s / avg_s * 2) + 1
        if target_s < 90:
            return max(3, n)
        return max(100, n)

    def hf_load_kwargs(self) -> Dict[str, Any]:
        return dict(self.extra_kwargs)

    def cls_label(self) -> Literal[-1, 1, 2]:
        match self.cls:
            case "speech": 
                return -1 
            case "music": 
                return 1
            case "inactive":
                 return 2

@dataclass
class TierConfig:
    entries: list[SourceEntry]





