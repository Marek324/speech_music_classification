"""
Dataset builder — 20 hours total (80/10/10 train/val/test)
============================================================

Macro split:
    Speech  45%  →  9h
    Music   45%  →  9h
    Noise   10%  →  2h

Speech sub-classes:
    train:    clean 40% | corrupted 20% | noisy 20% | speech+music 7.5% | multispeaker 7.5% | MS+music 5%
    val/test: clean 30% | corrupted 20% | noisy 20% | speech+music 10%  | multispeaker 10%  | MS+music 10%

Music sub-classes (equal shares, all splits):
    electronic | folk | hip-hop | instrumental | pop | rock  →  rpmon/fma-genre-classification
    jazz                                                     →  PlutoG99001/MusicGen-Jazz-Clean
    country                                                  →  ylacombe/music_genres_Country
    vocal (acappella)                                        →  ccmusic-database/acapella (song1/2/3)

Noise:
    AxonData/background-noise-detection-dataset  (primary, real field recordings)
    danavery/urbansound8K                        (supplementary variety)

Row counts are derived from avg clip duration to hit the target hours per class.
All splits are 80/10/10 of total_rows for each source.
"""

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

    total_rows        — rows consumed in total; split 80/10/10 internally
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
    total_rows: int
    detector: str  # "vad" | "music" | "silence"
    audio_col: str = "audio"
    sil_thr: int = -16
    split: str = "train"
    audio_decode: bool = True
    filter_col: str | None = None
    filter_val: str | None = None
    skip: int = 0
    extra_load_kwargs: dict = field(default_factory=dict)


# ── SPEECH ────────────────────────────────────────────────────────────────────
#
# Target hours (train / val / test):
#   clean          2.88h / 0.27h / 0.27h  — avg 12.6s/utt  → 822 rows total
#   corrupted      1.44h / 0.18h / 0.18h  — avg 12.0s/utt  → 432 rows total
#   noisy          1.44h / 0.18h / 0.18h  — avg  3.7s/utt  → 1401 rows total
#   speech+music   0.54h / 0.09h / 0.09h  — avg 10.0s/clip →  194 rows total
#   multispeaker   0.54h / 0.09h / 0.09h  — avg  5.0s/seg  →  389 rows total
#   MS+music       0.36h / 0.09h / 0.09h  — avg  5.0s/seg  →  260 rows total
#
SPEECH_SOURCES: list[DatasetSpec] = [
    # ── Clean ────────────────────────────────────────────────────────────
    # openslr/librispeech_asr  config "clean"  train.100
    # Studio-quality audiobook recordings, curated for low WER
    # https://huggingface.co/datasets/openslr/librispeech_asr
    DatasetSpec(
        name="clean_librispeech",
        hf_id="openslr/librispeech_asr",
        cls="speech",
        subclass="speech_clean",
        total_rows=822,
        detector="vad",
        split="train.100",
        audio_decode=True,
        extra_load_kwargs={"name": "clean"},
    ),
    # ── Corrupted ────────────────────────────────────────────────────────
    # openslr/librispeech_asr  config "other"  train.500
    # Lower-quality readers, inconsistent recording conditions, higher WER
    # https://huggingface.co/datasets/openslr/librispeech_asr
    DatasetSpec(
        name="corrupted_librispeech",
        hf_id="openslr/librispeech_asr",
        cls="speech",
        subclass="speech_corrupted",
        total_rows=432,
        detector="vad",
        split="train.500",
        audio_decode=True,
        extra_load_kwargs={"name": "other"},
    ),
    # ── Noisy — real environments ─────────────────────────────────────────
    # JacobLinCool/VoiceBank-DEMAND-16k
    # VCTK speech mixed with real DEMAND room recordings (office, restaurant, station…)
    # https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k
    DatasetSpec(
        name="noisy_voicebank_demand",
        hf_id="JacobLinCool/VoiceBank-DEMAND-16k",
        cls="speech",
        subclass="speech_noisy",
        total_rows=1401,
        audio_col="noisy",
        detector="vad",
        split="train",
        audio_decode=True,
    ),
    # ── Speech over music ────────────────────────────────────────────────
    # cristian-rodriguez/ava_speech  class "speech_with_music"
    # Real movie audio segments labelled by Google AVA-Speech
    # https://huggingface.co/datasets/cristian-rodriguez/ava_speech
    DatasetSpec(
        name="speech_over_music_ava",
        hf_id="cristian-rodriguez/ava_speech",
        cls="speech",
        subclass="speech_over_music",
        total_rows=194,
        detector="vad",
        split="train",
        audio_decode=True,
        filter_col="label",
        filter_val="speech_with_music",
    ),
    # ── Multi-speaker ────────────────────────────────────────────────────
    # edinburghcstr/ami  config "headset-single" (IHM)
    # Real English meeting recordings, 3-5 speakers, pre-segmented utterances
    # https://huggingface.co/datasets/edinburghcstr/ami
    DatasetSpec(
        name="multispeaker_ami_ihm",
        hf_id="edinburghcstr/ami",
        cls="speech",
        subclass="speech_multispeaker",
        total_rows=389,
        detector="vad",
        split="train",
        audio_decode=True,
        extra_load_kwargs={"name": "headset-single"},
    ),
    # ── Multi-speaker over music ─────────────────────────────────────────
    # edinburghcstr/ami  config "microphone-single" (SDM / far-field)
    # Far-field mic captures natural speaker overlap and room bleed.
    # NOTE: after writing, mix each output wav with a random FMA clip at ~-10 dB
    #       (e.g. audiomentations.AddBackgroundNoise) to produce the final class.
    # https://huggingface.co/datasets/edinburghcstr/ami
    DatasetSpec(
        name="multispeaker_over_music_ami_sdm",
        hf_id="edinburghcstr/ami",
        cls="speech",
        subclass="speech_multispeaker_over_music",
        total_rows=260,
        detector="vad",
        split="train",
        audio_decode=True,
        extra_load_kwargs={"name": "microphone-single"},
    ),
]


# ── MUSIC ─────────────────────────────────────────────────────────────────────
#
# Target per genre: 0.90h train / 0.1125h val / 0.1125h test
#   FMA clips  30s → 108 rows  (108 × 30s = 3240s = 0.90h)
#   GTZAN      30s → 100 rows available, capped: 80 train / 10 val / 10 test
#   Acappella ~90s → 36 rows split across 3 song splits (12 each)
#
MUSIC_SOURCES: list[DatasetSpec] = [
    DatasetSpec(
        name="music_electronic",
        hf_id="rpmon/fma-genre-classification",
        cls="music",
        subclass="music_electronic",
        total_rows=108,
        detector="music",
        sil_thr=-25,
        split="train",
        audio_decode=False,
        filter_col="label",
        filter_val="Electronic",
    ),
    DatasetSpec(
        name="music_folk",
        hf_id="rpmon/fma-genre-classification",
        cls="music",
        subclass="music_folk",
        total_rows=108,
        detector="music",
        sil_thr=-33,
        split="train",
        audio_decode=False,
        filter_col="label",
        filter_val="Folk",
    ),
    DatasetSpec(
        name="music_hiphop",
        hf_id="rpmon/fma-genre-classification",
        cls="music",
        subclass="music_hiphop",
        total_rows=108,
        detector="music",
        sil_thr=-22,
        split="train",
        audio_decode=False,
        filter_col="label",
        filter_val="Hip-Hop",
    ),
    DatasetSpec(
        name="music_instrumental",
        hf_id="rpmon/fma-genre-classification",
        cls="music",
        subclass="music_instrumental",
        total_rows=108,
        detector="music",
        sil_thr=-27,
        split="train",
        audio_decode=False,
        filter_col="label",
        filter_val="Instrumental",
    ),
    DatasetSpec(
        name="music_pop",
        hf_id="rpmon/fma-genre-classification",
        cls="music",
        subclass="music_pop",
        total_rows=108,
        detector="music",
        sil_thr=-22,
        split="train",
        audio_decode=False,
        filter_col="label",
        filter_val="Pop",
    ),
    DatasetSpec(
        name="music_rock",
        hf_id="rpmon/fma-genre-classification",
        cls="music",
        subclass="music_rock",
        total_rows=108,
        detector="music",
        sil_thr=-30,
        split="train",
        audio_decode=False,
        filter_col="label",
        filter_val="Rock",
    ),
    DatasetSpec(
        name="music_jazz",
        hf_id="PlutoG99001/MusicGen-Jazz-Clean",
        cls="music",
        subclass="music_jazz",
        total_rows=100,
        detector="music",
        sil_thr=-20,
        split="train",
        audio_decode=False,
        filter_col="genre",
        filter_val="jazz",
        extra_load_kwargs={"name": "all"},
    ),
    DatasetSpec(
        name="music_country",
        hf_id="ylacombe/music_genres_Country",
        cls="music",
        subclass="music_country",
        total_rows=100,
        detector="music",
        sil_thr=-25,
        split="train",
        audio_decode=False,
        filter_col="genre",
        filter_val="country",
        extra_load_kwargs={"name": "all"},
    ),
    # ── Vocal / acappella ────────────────────────────────────────────────
    # ccmusic-database/acapella  splits song1 / song2 / song3
    # ~90s clips, pure singing with no instrumental background
    # 12 rows × 3 splits × ~90s ≈ 54min = 0.90h train target
    # https://huggingface.co/datasets/ccmusic-database/acapella
    DatasetSpec(
        name="music_vocal_song1",
        hf_id="ccmusic-database/acapella",
        cls="music",
        subclass="music_vocal",
        total_rows=12,
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
        total_rows=12,
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
        total_rows=12,
        detector="music",
        sil_thr=-48,
        split="song3",
        audio_decode=False,
    ),
]


# ── NOISE ─────────────────────────────────────────────────────────────────────
#
# Target: 1.6h train / 0.2h val / 0.2h test
#   AxonData clips are continuous field recordings — entire clips labelled "inactive"
#   UrbanSound8K clips ≤ 4s avg ~3.5s — adds class variety
#
NOISE_SOURCES: list[DatasetSpec] = [
    # Primary — AxonData/background-noise-detection-dataset
    # 50+ hours of real non-synthetic field recordings: airport, street, subway
    # https://huggingface.co/datasets/AxonData/background-noise-detection-dataset
    DatasetSpec(
        name="noise_axon",
        hf_id="AxonData/background-noise-detection-dataset",
        cls="noise",
        subclass="noise",
        total_rows=576,
        detector="silence",
        split="train",
        audio_decode=True,
    ),
    # Supplementary — danavery/urbansound8K
    # 8732 clips ≤ 4s across 10 urban sound classes, adds variety
    # https://huggingface.co/datasets/danavery/urbansound8K
    DatasetSpec(
        name="noise_urbansound",
        hf_id="danavery/urbansound8K",
        cls="noise",
        subclass="noise",
        total_rows=250,
        detector="silence",
        split="train",
        audio_decode=True,
    ),
]


ALL_SOURCES: list[DatasetSpec] = SPEECH_SOURCES + MUSIC_SOURCES + NOISE_SOURCES


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
    """Stream, filter, shuffle, take, and cast one source into a ready IterableDataset."""
    print(f"  Loading {spec.name}  ({spec.hf_id}) ...")

    ds = load_dataset(
        spec.hf_id,
        split=spec.split,
        streaming=True,
        **spec.extra_load_kwargs,
    )
    assert isinstance(ds, IterableDataset)

    ds = ds.shuffle(seed=RAND_SEED)

    if spec.filter_col and spec.filter_val:
        ds = ds.filter(lambda row: row[spec.filter_col] == spec.filter_val)

    if spec.skip > 0:
        ds = ds.skip(spec.skip)

    ds = ds.take(spec.total_rows)
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
    """Load one source and write train / val / test splits to disk."""
    print(f"\n{'─' * 60}")
    print(f"  {spec.name}  [{spec.cls} → {spec.subclass}]  ({spec.total_rows} rows)")
    print(f"{'─' * 60}")

    ds = load_source(spec)
    labeler = get_labeler(spec.detector)

    train_end = int(0.8 * spec.total_rows)
    val_end = int(0.9 * spec.total_rows)

    splits = [
        ("train", ds.take(train_end)),
        ("val", ds.skip(train_end).take(val_end - train_end)),
        ("test", ds.skip(val_end)),
    ]

    for split_name, split_data in splits:
        writer = SplitWriter(split_name)
        local_idx = 0

        for row in tqdm(split_data, desc=f"    {split_name}"):
            labels = labeler.label(row[spec.audio_col], spec.sil_thr)

            if labels is None:
                print(
                    f"  [SKIP] {spec.name} row {local_idx} — invalid audio",
                    file=sys.stderr,
                )
                continue

            writer.write(
                row[spec.audio_col], labels, spec.name, local_idx, spec.cls, spec.subclass
            )
            local_idx += 1

    print(f"  Done: {spec.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Dataset build started")
    print(f"Output : {DATA_DIR.resolve()}")
    print(f"Sources: {len(ALL_SOURCES)}")
    print("=" * 60)

    for spec in ALL_SOURCES:
        process_source(spec)

    print("\n" + "=" * 60)
    print("Build complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
