"""
Dataset builder — configurable total hours (80/10/10 train/val/test)
====================================================================

Usage:
    uv run python build.py mini   # 1h — quick sanity check
    uv run python build.py mid    # 20h — matches original dataset
    uv run python build.py full   # 200h — maximum scale

Macro split:
    Speech  45%
    Music   45%
    Noise   10%

Each source has a target_seconds budget. Clips are streamed until the time
budget is filled (not a fixed row count), so actual hours match the target
regardless of per-clip duration variance.

All splits are 80/10/10 of target_seconds for each source.
"""

import argparse
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jsonlines
import soundfile as sf
from datasets import Audio, IterableDataset, load_dataset
from pydub import AudioSegment, silence
from silero_vad import get_speech_timestamps, load_silero_vad
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SIZE_PRESETS = {"mini": 1, "mid": 20, "full": 200}

SPEECH_FRAC = 0.45
MUSIC_FRAC = 0.45
NOISE_FRAC = 0.10

RAND_SEED = 381
DATA_DIR = Path("data")
MAX_FILES_PER_FOLDER = 9000
SR = 16000


# ─────────────────────────────────────────────────────────────────────────────
# Dataset specs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetSpec:
    """
    Fully describes one source dataset and how to process it.

    target_seconds    — total audio duration to collect; split 80/10/10 internally
    max_rows          — hard cap on rows (for small/finite sources); 0 = unlimited
    detector          — "vad" | "music" | "silence"
    audio_decode      — True  → decoded AudioDecoder  (required by VADLabeler)
                        False → raw bytes dict        (required by MusicLabeler)
    filter_col/val    — optional single-column row filter (e.g. genre == "jazz")
    skip              — skip N rows before taking (for multiple chunks from one dataset)
    extra_load_kwargs — forwarded verbatim to load_dataset()
    """

    name: str
    hf_id: str
    cls: str  # "speech" | "music" | "noise"
    subclass: str  # e.g. "speech_clean", "music_jazz"
    target_seconds: float
    detector: str  # "vad" | "music" | "silence"
    max_rows: int = 0
    audio_col: str = "audio"
    sil_thr: int = -16
    split: str = "train"
    audio_decode: bool = True
    filter_col: str | None = None
    filter_val: int | None = None
    skip: int = 0
    extra_load_kwargs: dict = field(default_factory=dict)


def build_sources(total_hours: int) -> tuple[list[DatasetSpec], list[DatasetSpec], list[DatasetSpec]]:
    """Build source lists scaled to *total_hours*."""
    speech_s = total_hours * SPEECH_FRAC * 3600
    music_s = total_hours * MUSIC_FRAC * 3600
    noise_s = total_hours * NOISE_FRAC * 3600

    # ── SPEECH ────────────────────────────────────────────────────────────
    # Share weights (sum to ~1.0 for non-synthetic sources)
    _speech_shares = {
        "ls100": 0.227,       # LibriSpeech train.100 — 20h @ 200h
        "ls360": 0.278,       # LibriSpeech train.360 — 24.5h @ 200h
        "ls_other": 0.162,    # LibriSpeech other — 14.3h @ 200h
        "voicebank": 0.105,   # VoiceBank-DEMAND — 9.3h @ 200h
        "ami": 0.126,         # AMI IHM — 11.1h @ 200h
        # som/msom handled by augment_over_music.py
    }

    speech_sources = [
        DatasetSpec(
            name="clean_librispeech",
            hf_id="openslr/librispeech_asr",
            cls="speech",
            subclass="speech_clean",
            target_seconds=speech_s * _speech_shares["ls100"],
            detector="vad",
            split="train.100",
            audio_decode=True,
            extra_load_kwargs={"name": "clean"},
        ),
        DatasetSpec(
            name="clean_librispeech_360",
            hf_id="openslr/librispeech_asr",
            cls="speech",
            subclass="speech_clean",
            target_seconds=speech_s * _speech_shares["ls360"],
            detector="vad",
            split="train.360",
            audio_decode=True,
            extra_load_kwargs={"name": "clean"},
        ),
        DatasetSpec(
            name="corrupted_librispeech",
            hf_id="openslr/librispeech_asr",
            cls="speech",
            subclass="speech_corrupted",
            target_seconds=speech_s * _speech_shares["ls_other"],
            detector="vad",
            split="train.500",
            audio_decode=True,
            extra_load_kwargs={"name": "other"},
        ),
        DatasetSpec(
            name="noisy_voicebank_demand",
            hf_id="JacobLinCool/VoiceBank-DEMAND-16k",
            cls="speech",
            subclass="speech_noisy",
            target_seconds=speech_s * _speech_shares["voicebank"],
            audio_col="noisy",
            detector="vad",
            split="train",
            audio_decode=True,
        ),
        DatasetSpec(
            name="multispeaker_ami_ihm",
            hf_id="edinburghcstr/ami",
            cls="speech",
            subclass="speech_multispeaker",
            target_seconds=speech_s * _speech_shares["ami"],
            detector="vad",
            split="train",
            audio_decode=True,
            extra_load_kwargs={"name": "ihm"},
        ),
    ]

    # ── MUSIC ─────────────────────────────────────────────────────────────
    fma_genre_s = music_s * 0.44 / 6
    fma_genres = [
        ("music_electronic", 0, -25),
        ("music_folk", 2, -33),
        ("music_hiphop", 3, -22),
        ("music_instrumental", 4, -27),
        ("music_pop", 6, -22),
        ("music_rock", 7, -30),
    ]

    music_sources: list[DatasetSpec] = [
        DatasetSpec(
            name=name,
            hf_id="rpmon/fma-genre-classification",
            cls="music",
            subclass=name,
            target_seconds=fma_genre_s,
            detector="music",
            sil_thr=sil,
            split="train",
            audio_decode=False,
            filter_col="genre",
            filter_val=gid,
        )
        for name, gid, sil in fma_genres
    ]

    # Small sources — capped at dataset max, scaled down at low hours
    small_share_s = music_s * 0.009
    vocal_share_s = music_s * 0.01 / 3

    music_sources += [
        DatasetSpec(
            name="music_jazz",
            hf_id="PlutoG99001/MusicGen-Jazz-Clean",
            cls="music",
            subclass="music_jazz",
            target_seconds=small_share_s,
            max_rows=100,
            detector="music",
            sil_thr=-20,
            split="train",
            audio_decode=False,
        ),
        DatasetSpec(
            name="music_country",
            hf_id="ylacombe/music_genres_Country",
            cls="music",
            subclass="music_country",
            target_seconds=small_share_s,
            max_rows=100,
            detector="music",
            sil_thr=-25,
            split="train",
            audio_decode=False,
        ),
        DatasetSpec(
            name="music_vocal_song1",
            hf_id="ccmusic-database/acapella",
            cls="music",
            subclass="music_vocal",
            target_seconds=vocal_share_s,
            max_rows=12,
            detector="music",
            sil_thr=-48,
            split="song1",
            audio_decode=False,
        ),
        DatasetSpec(
            name="music_vocal_song2",
            hf_id="ccmusic-database/acapella",
            cls="music",
            subclass="music_vocal",
            target_seconds=vocal_share_s,
            max_rows=12,
            detector="music",
            sil_thr=-48,
            split="song2",
            audio_decode=False,
        ),
        DatasetSpec(
            name="music_vocal_song3",
            hf_id="ccmusic-database/acapella",
            cls="music",
            subclass="music_vocal",
            target_seconds=vocal_share_s,
            max_rows=12,
            detector="music",
            sil_thr=-48,
            split="song3",
            audio_decode=False,
        ),
    ]

    # GTZAN — 999 clips available
    music_sources.append(
        DatasetSpec(
            name="music_gtzan",
            hf_id="marsyas/gtzan",
            cls="music",
            subclass="music_gtzan",
            target_seconds=music_s * 0.09,
            max_rows=999,
            detector="music",
            sil_thr=-25,
            split="train",
            audio_decode=False,
        ),
    )

    # MTG-Jamendo — fill remaining music hours
    fma_total_s = fma_genre_s * 6
    gtzan_s = music_s * 0.09
    fixed_s = small_share_s * 2 + vocal_share_s * 3
    jamendo_s = max(0, music_s - fma_total_s - gtzan_s - fixed_s)
    if jamendo_s > 0:
        music_sources.append(
            DatasetSpec(
                name="music_jamendo",
                hf_id="rkstgr/mtg-jamendo",
                cls="music",
                subclass="music_jamendo",
                target_seconds=jamendo_s,
                detector="music",
                sil_thr=-25,
                split="train",
                audio_decode=False,
            ),
        )

    # ── NOISE ─────────────────────────────────────────────────────────────
    noise_sources = [
        DatasetSpec(
            name="noise_axon",
            hf_id="AxonData/background-noise-detection-dataset",
            cls="noise",
            subclass="noise",
            target_seconds=noise_s * 0.80,
            detector="silence",
            split="train",
            audio_decode=True,
        ),
        DatasetSpec(
            name="noise_urbansound",
            hf_id="danavery/urbansound8K",
            cls="noise",
            subclass="noise",
            target_seconds=noise_s * 0.06,
            detector="silence",
            split="train",
            audio_decode=True,
        ),
        DatasetSpec(
            name="noise_fsd50k",
            hf_id="Fhrozen/FSD50k",
            cls="noise",
            subclass="noise_environmental",
            target_seconds=noise_s * 0.14,
            detector="silence",
            split="train",
            audio_decode=True,
        ),
    ]

    return speech_sources, music_sources, noise_sources




# ─────────────────────────────────────────────────────────────────────────────
# Duration helpers
# ─────────────────────────────────────────────────────────────────────────────


def clip_duration_s(audio: AudioDecoder | dict) -> float:
    """Return clip duration in seconds."""
    if isinstance(audio, AudioDecoder):
        return audio.get_all_samples().duration_seconds
    seg = (
        AudioSegment.from_file(io.BytesIO(audio["bytes"]))
        if audio["bytes"] is not None
        else AudioSegment.from_file(audio["path"])
    )
    return len(seg) / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Labelers
# ─────────────────────────────────────────────────────────────────────────────


def ms(samples: int) -> int:
    """Convert sample count → milliseconds at global SR."""
    return int(samples * 1000 / SR)


def make_label(label: str, start: int, end: int) -> dict:
    return {"label": label, "start": start, "end": end}


class VADLabeler:
    """
    Silero VAD → speech / inactive labels.
    Requires audio_decode=True (AudioDecoder input).
    """

    def __init__(self):
        self.model = load_silero_vad()

    def label(self, audio: AudioDecoder, _thr: int) -> list[dict] | None:
        assert isinstance(audio, AudioDecoder)
        try:
            samples = audio.get_all_samples()
        except RuntimeError:
            return None

        timestamps = get_speech_timestamps(samples.data, self.model)

        if not timestamps:
            return [make_label("inactive", 0, ms(len(samples.data)))]

        labels = []
        prev_end = 0
        for ts in timestamps:
            if ts["start"] > prev_end:
                labels.append(make_label("inactive", ms(prev_end), ms(ts["start"])))
            labels.append(make_label("speech", ms(ts["start"]), ms(ts["end"])))
            prev_end = ts["end"]

        if prev_end < len(samples.data):
            labels.append(make_label("inactive", ms(prev_end), ms(len(samples.data))))

        return labels


class MusicLabeler:
    """
    pydub silence detection → music / inactive labels.
    Requires audio_decode=False (raw bytes dict input).
    """

    def label(self, audio: dict, silence_threshold: int) -> list[dict] | None:
        assert isinstance(audio, dict)
        seg = (
            AudioSegment.from_file(io.BytesIO(audio["bytes"]))
            if audio["bytes"] is not None
            else AudioSegment.from_file(audio["path"])
        )

        silent_ranges = silence.detect_silence(
            seg, min_silence_len=260, silence_thresh=silence_threshold
        )

        if not silent_ranges:
            return [make_label("music", 0, len(seg))]

        labels = []
        prev_end = 0
        for sil_start, sil_end in silent_ranges:
            if sil_start > prev_end:
                labels.append(make_label("music", prev_end, sil_start))
            labels.append(make_label("inactive", sil_start, sil_end))
            prev_end = sil_end

        if prev_end < len(seg):
            labels.append(make_label("music", prev_end, len(seg)))

        return labels


class SilenceLabeler:
    """
    Labels the entire clip as inactive.
    Used for noise files — no further segmentation needed.
    Accepts either AudioDecoder or raw bytes dict.
    """

    def label(self, audio: AudioDecoder | dict, _thr: int) -> list[dict]:
        if isinstance(audio, AudioDecoder):
            duration_ms = int(audio.get_all_samples().duration_seconds * 1000)
        else:
            seg = (
                AudioSegment.from_file(io.BytesIO(audio["bytes"]))
                if audio["bytes"] is not None
                else AudioSegment.from_file(audio["path"])
            )
            duration_ms = len(seg)
        return [make_label("inactive", 0, duration_ms)]


_LABELER_CACHE: dict = {}


def get_labeler(detector: str) -> VADLabeler | MusicLabeler | SilenceLabeler:
    """Lazily instantiate and cache labelers (VAD model load is expensive)."""
    if detector not in _LABELER_CACHE:
        match detector:
            case "vad":
                _LABELER_CACHE["vad"] = VADLabeler()
            case "music":
                _LABELER_CACHE["music"] = MusicLabeler()
            case "silence":
                _LABELER_CACHE["silence"] = SilenceLabeler()
            case _:
                raise ValueError(f"Unknown detector: {detector!r}")
    return _LABELER_CACHE[detector]


# ─────────────────────────────────────────────────────────────────────────────
# File writer
# ─────────────────────────────────────────────────────────────────────────────


class SplitWriter:
    """
    Writes .wav + metadata.jsonl into  data/{split}/{folder_idx:03d}/
    Rotates to a new subfolder every MAX_FILES_PER_FOLDER files.
    Safe to resume: skips already-written files on restart.
    """

    def __init__(self, split: str):
        self.base = DATA_DIR / split
        self.base.mkdir(parents=True, exist_ok=True)
        self._resume()

    def _resume(self):
        existing = sorted(
            [d for d in self.base.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if not existing:
            self.folder_idx = 0
            self.folder = self.base / "000"
            self.folder.mkdir()
            self.file_count = 0
        else:
            self.folder = existing[-1]
            self.folder_idx = int(self.folder.name)
            self.file_count = len(list(self.folder.glob("*.wav")))

    def _rotate_if_full(self):
        if self.file_count >= MAX_FILES_PER_FOLDER:
            self.folder_idx += 1
            self.folder = self.base / f"{self.folder_idx:03d}"
            self.folder.mkdir(exist_ok=True)
            self.file_count = 0

    def write(
        self,
        audio: AudioDecoder | dict,
        labels: list[dict],
        name: str,
        idx: int,
        cls: str,
        subclass: str,
    ):
        self._rotate_if_full()

        file_name = f"{name}_{idx:05d}.wav"
        file_path = self.folder / file_name

        if file_path.exists():
            return  # already written — safe resume

        if isinstance(audio, AudioDecoder):
            data = audio.get_all_samples().data
            if hasattr(data, "cpu"):
                data = data.cpu()
            sf.write(file_path, data.numpy().squeeze(), samplerate=SR)
        else:
            seg = (
                AudioSegment.from_file(io.BytesIO(audio["bytes"]))
                if audio["bytes"] is not None
                else AudioSegment.from_file(audio["path"])
            )
            seg.export(file_path, format="wav")

        with jsonlines.open(self.folder / "metadata.jsonl", mode="a") as f:
            f.write(
                {
                    "file_name": file_name,
                    "class": cls,
                    "subclass": subclass,
                    "labels": labels,
                }
            )

        self.file_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────


def load_source(spec: DatasetSpec) -> IterableDataset:
    """Stream, filter, shuffle, and cast one source into a ready IterableDataset.

    Unlike the row-based version, this does NOT call .take() — the caller
    (process_source) stops consuming once the time budget is filled.
    If max_rows is set, the stream is capped at that many rows.
    """
    print(f"  Loading {spec.name}  ({spec.hf_id}) ...")

    ds = load_dataset(
        spec.hf_id,
        split=spec.split,
        streaming=True,
        **spec.extra_load_kwargs,
    )
    assert isinstance(ds, IterableDataset)

    ds = ds.shuffle(seed=RAND_SEED)

    if spec.filter_col is not None and spec.filter_val is not None:
        ds = ds.filter(lambda row: row[spec.filter_col] == spec.filter_val)

    if spec.skip > 0:
        ds = ds.skip(spec.skip)

    if spec.max_rows > 0:
        ds = ds.take(spec.max_rows)

    ds = ds.select_columns(spec.audio_col)
    ds = ds.cast_column(
        spec.audio_col,
        Audio(sampling_rate=SR, num_channels=1, decode=spec.audio_decode),
    )
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────


def process_source(spec: DatasetSpec):
    """Load one source and write train / val / test splits to disk.

    Clips are consumed from the stream and assigned to splits in order:
    train (80%), val (10%), test (10%) — based on accumulated *seconds*,
    not row count.
    """
    target_s = spec.target_seconds
    print(f"\n{'─' * 60}")
    print(f"  {spec.name}  [{spec.cls} → {spec.subclass}]  (target {target_s / 3600:.2f}h)")
    print(f"{'─' * 60}")

    ds = load_source(spec)
    labeler = get_labeler(spec.detector)

    # Time budgets per split
    budgets = {
        "train": target_s * 0.80,
        "val": target_s * 0.10,
        "test": target_s * 0.10,
    }
    accumulated = {"train": 0.0, "val": 0.0, "test": 0.0}
    counts = {"train": 0, "val": 0, "test": 0}
    split_order = ["train", "val", "test"]
    current_split_idx = 0

    writers = {s: SplitWriter(s) for s in split_order}
    skipped = 0

    for row in tqdm(ds, desc=f"    {spec.name}"):
        # Determine which split this clip goes to
        while current_split_idx < len(split_order):
            split_name = split_order[current_split_idx]
            if accumulated[split_name] < budgets[split_name]:
                break
            current_split_idx += 1
        else:
            break  # all splits filled

        split_name = split_order[current_split_idx]
        audio = row[spec.audio_col]

        labels = labeler.label(audio, spec.sil_thr)
        if labels is None:
            skipped += 1
            continue

        dur = clip_duration_s(audio)
        writers[split_name].write(
            audio, labels, spec.name, counts[split_name], spec.cls, spec.subclass
        )
        accumulated[split_name] += dur
        counts[split_name] += 1

    total_s = sum(accumulated.values())
    total_clips = sum(counts.values())
    print(f"  Done: {spec.name}  — {total_s / 3600:.2f}h ({total_clips} clips)", end="")
    for s in split_order:
        print(f"  {s}={accumulated[s] / 3600:.2f}h({counts[s]})", end="")
    if skipped:
        print(f"  skipped={skipped}", end="")
    print()

    if total_s < target_s * 0.9:
        print(
            f"  [WARN] Only collected {total_s / 3600:.2f}h of {target_s / 3600:.2f}h target"
            f" — source may have fewer clips than needed",
            file=sys.stderr,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Parquet export
# ─────────────────────────────────────────────────────────────────────────────


def export_parquet():
    """Read data/{train,val,test}/ WAVs + metadata.jsonl → parquet files."""
    from datasets import Audio as AudioFeature, Dataset, DatasetDict, Features, Value

    splits = {}
    for split_name in ("train", "val", "test"):
        split_dir = DATA_DIR / split_name
        if not split_dir.exists():
            continue

        rows: list[dict] = []
        for folder in sorted(split_dir.iterdir()):
            meta_path = folder / "metadata.jsonl"
            if not meta_path.exists():
                continue
            with jsonlines.open(meta_path) as reader:
                for entry in reader:
                    wav_path = str((folder / entry["file_name"]).resolve())
                    rows.append(
                        {
                            "audio": wav_path,
                            "class": entry["class"],
                            "subclass": entry["subclass"],
                            "labels": entry["labels"],
                        }
                    )

        if not rows:
            continue

        ds = Dataset.from_dict(
            {k: [r[k] for r in rows] for k in rows[0]},
        ).cast_column("audio", AudioFeature(sampling_rate=SR))
        splits[split_name] = ds
        print(f"  {split_name}: {len(ds)} clips")

    dd = DatasetDict(splits)
    out_dir = DATA_DIR / "parquet"
    dd.save_to_disk(str(out_dir))
    print(f"  Saved to {out_dir.resolve()}")

    # Also write individual .parquet files for easy HF upload
    for split_name, ds in dd.items():
        pq_path = DATA_DIR / "parquet" / f"{split_name}.parquet"
        ds.to_parquet(str(pq_path))
        print(f"  {pq_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HF upload
# ─────────────────────────────────────────────────────────────────────────────

HF_REPO = "Marek324/speech-music-classification"


def upload_to_hf(revision: str):
    """Upload data/parquet/ to HF as a dataset revision (branch)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    api.create_branch(HF_REPO, repo_type="dataset", branch=revision, exist_ok=True)

    parquet_dir = DATA_DIR / "parquet"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"No parquet directory at {parquet_dir}. Run export first.")

    print(f"  Uploading {parquet_dir} → {HF_REPO} (revision={revision})")
    api.upload_large_folder(
        repo_id=HF_REPO,
        folder_path=str(parquet_dir),
        repo_type="dataset",
        revision=revision,
    )
    print(f"  Done: https://huggingface.co/datasets/{HF_REPO}/tree/{revision}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build speech/music/noise dataset")
    parser.add_argument(
        "size",
        choices=SIZE_PRESETS,
        help="mini (1h), mid (20h), or full (200h)",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload parquet to HuggingFace after build",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_hours = SIZE_PRESETS[args.size]

    speech_sources, music_sources, noise_sources = build_sources(total_hours)
    all_sources = speech_sources + music_sources + noise_sources

    print("=" * 60)
    print(f"Dataset build started  ({total_hours}h target — {args.size})")
    print(f"  Speech {total_hours * SPEECH_FRAC:.0f}h | Music {total_hours * MUSIC_FRAC:.0f}h | Noise {total_hours * NOISE_FRAC:.0f}h")
    print(f"Output : {DATA_DIR.resolve()}")
    print(f"Sources: {len(all_sources)}")
    print("=" * 60)

    for spec in all_sources:
        process_source(spec)

    print("\n" + "=" * 60)
    print("Downloads complete — starting synthetic augmentation...")
    print("=" * 60)

    from augment_over_music import run_augmentation
    run_augmentation(total_hours)

    print("\n" + "=" * 60)
    print("Exporting to parquet...")
    print("=" * 60)
    export_parquet()

    if args.upload:
        print("\n" + "=" * 60)
        print(f"Uploading to HuggingFace as revision '{args.size}'...")
        print("=" * 60)
        upload_to_hf(args.size)

    print("\n" + "=" * 60)
    print("Build complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
