import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import jsonlines
import soundfile as sf
from datasets import Audio, Dataset, IterableDataset, load_dataset, load_from_disk
from pydub import AudioSegment, silence
from silero_vad import get_speech_timestamps, load_silero_vad
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

RAND_SEED = 381
DATA_DIR = "data"

NAMES = [
    "clean1",
    "clean2",
    "clean3",
    "clean4",
    "clean5",
    "clean6",
    "clean7",
    "clean8",
    "clean9",
    "noisy",
    "noisyenv",
    "noise",
    "jazz",
    "country",
    "folk",
    "pop",
    "rock",
    "electronic",
    "instrumental",
    "vocal1",
    "vocal2",
    "vocal3",
]

sr = 16000
fl = 20 * sr // 1000
fh = 10 * sr // 1000


class Silence:
    def __init__(self): ...

    def label(
        self, x: AudioDecoder | dict[str, str | bytes], _
    ) -> Optional[list[dict[str, dict[str, int]]]]:
        if isinstance(x, AudioDecoder):
            siglen = int(x.get_all_samples().duration_seconds * 1000)
        else:
            if x["bytes"] is not None:
                seg = AudioSegment.from_file(io.BytesIO(x["bytes"]))
            else:
                seg = AudioSegment.from_file(x["path"])

            siglen = len(seg)

        return [{"inactive": {"start": 0, "end": siglen}}]


class VAD:
    def __init__(self):
        self.model = load_silero_vad()

    def label(
        self, decoder: AudioDecoder | dict[str, str | bytes], _
    ) -> Optional[list[dict[str, dict[str, int]]]]:
        def stoms(sample: int) -> int:
            # sample to ms
            global sr
            return int(sample * 1000 / sr)

        assert isinstance(decoder, AudioDecoder)

        try:
            x = decoder.get_all_samples()
        except RuntimeError as e:
            return None

        stamps: list[dict[str, int]] = get_speech_timestamps(x.data, self.model)

        if not stamps:
            return [{"inactive": {"start": 0, "end": len(x.data)}}]

        labels: list[dict[str, dict[str, int]]] = []
        prev_end = 0

        for stamp in stamps:
            if stamp["start"] > prev_end:
                labels.append(
                    {
                        "inactive": {
                            "start": stoms(prev_end),
                            "end": stoms(stamp["start"]),
                        }
                    }
                )

            labels.append(
                {"speech": {"start": stoms(stamp["start"]), "end": stoms(stamp["end"])}}
            )
            prev_end = stamp["end"]

        if prev_end < len(x.data):
            labels.append(
                {"inactive": {"start": stoms(prev_end), "end": stoms(len(x.data))}}
            )

        return labels


class MusicSilenceDetector:
    def __init__(self): ...

    def label(
        self, x: AudioDecoder | dict[str, str | bytes], thr: int
    ) -> Optional[list[dict[str, dict[str, int]]]]:
        assert isinstance(x, dict)
        labels: list[dict[str, dict[str, int]]] = []

        if x["bytes"] is not None:
            seg = AudioSegment.from_file(io.BytesIO(x["bytes"]))
        else:
            seg = AudioSegment.from_file(x["path"])

        sil_stamps: list[list[int]] = silence.detect_silence(
            seg, min_silence_len=260, silence_thresh=thr
        )  # min_silence_len is in ms

        if not sil_stamps:
            return [{"music": {"start": 0, "end": len(seg)}}]

        prev_end = 0

        for stamp in sil_stamps:
            if stamp[0] > prev_end:
                labels.append(
                    {
                        "music": {
                            "start": prev_end,
                            "end": stamp[0],
                        }
                    }
                )

            labels.append({"inactive": {"start": stamp[0], "end": stamp[1]}})
            prev_end = stamp[1]

        if prev_end < len(seg):
            labels.append({"music": {"start": prev_end, "end": len(seg)}})

        return labels


def process_file(
    audio: AudioDecoder | dict[str, str | bytes],
    silence_detector: VAD | MusicSilenceDetector | Silence,
    split: str,
    idx: int,
    name: str,
    sil_thr: int = -16,
) -> Optional[tuple[bool, list[dict[str, dict[str, int]]]]]:
    labels = silence_detector.label(audio, sil_thr)
    if labels is None:
        return None

    file_name = f"{DATA_DIR}/{split}/{name}_{idx:05d}.wav"
    if os.path.exists(file_name):
        return (False, labels)

    if isinstance(audio, AudioDecoder):
        audio_data = audio.get_all_samples().data
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu()
        audio_data = audio_data.numpy().squeeze()

        sf.write(
            file_name,
            audio_data,
            samplerate=sr,
        )
    else:
        seg = AudioSegment.from_file(io.BytesIO(audio["bytes"]))
        seg.export(file_name, format="wav")

    return (True, labels)


class Meta:
    def __init__(self, _name, _split: str):
        self.split = _split
        self.name = _name
        _dir = f"partial/{self.name}"
        os.makedirs(_dir, exist_ok=True)
        self.path = f"{_dir}/{self.split}.jsonl"
        self.file = jsonlines.open(self.path, mode="w")
        self.idx = 0

    def __del__(self):
        self.file.close()

    def write(
        self, _class: str, subclass: str, labels: list[dict[str, dict[str, int]]]
    ):
        self.file.write(
            {
                "file_name": f"{self.name}_{self.idx:05d}.wav",
                "class": _class,
                "subclass": subclass,
                "labels": labels,
            }
        )
        self.idx += 1


def process_dataset(
    dataset: IterableDataset,
    silence_detector: VAD | MusicSilenceDetector | Silence,
    _class: str,
    subclass: str,
    dataset_len: int,
    name: str,
    sil_thr: int = -16,
):
    print(f"Start processing {name}...")
    train_end = int(0.8 * dataset_len)
    val_end = int(0.9 * dataset_len)
    splits = [
        (dataset.take(train_end), Meta(name, "train")),
        (dataset.skip(train_end).take(val_end - train_end), Meta(name, "val")),
        (dataset.skip(val_end), Meta(name, "test")),
    ]

    for split, meta in splits:
        for row in tqdm(split, desc=f"{name} - {meta.split}"):
            res = process_file(
                row["audio"], silence_detector, meta.split, meta.idx, name, sil_thr
            )
            if res is None:
                print("Invalid data, skipping", file=sys.stderr)
                continue
            new, labels = res
            meta.write(_class, subclass, labels)

    print("Finished\n")


def combine_metas():
    for split in ["train", "val", "test"]:
        file_path = f"{DATA_DIR}/{split}/metadata.jsonl"
        with jsonlines.open(file_path, mode="w") as meta:
            for name in NAMES:
                try:
                    with jsonlines.open(f"partial/{name}/{split}.jsonl") as part:
                        for line in part:
                            meta.write(line)

                    print(f"Written to: {file_path}")
                except FileNotFoundError:
                    if split == "train":
                        print(f"{name} not downloaded")


def split_files(max_per_folder=9000):
    data_path = Path(DATA_DIR)
    for split in ["train", "test", "val"]:
        split_path = data_path / split
        if not split_path.exists():
            continue

        print(f"Processing {split} split...")

        # Get all files except metadata.jsonl
        all_files = [f for f in split_path.iterdir() if f.is_file()]
        audio_files = [f for f in all_files if f.name != "metadata.jsonl"]
        metadata_files = [f for f in all_files if f.name == "metadata.jsonl"]

        if len(audio_files) <= max_per_folder:
            print(f"  {split} has {len(audio_files)} files, no splitting needed")
            continue

        # Load original metadata
        original_metadata = {}
        if metadata_files:
            metadata_path = metadata_files[0]
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        # Use filename as key (adjust based on your metadata structure)
                        file_name = item.get(
                            "file_name", item.get("audio_file_path", "")
                        )
                        if file_name:
                            original_metadata[file_name] = item

        # Create numbered subfolders
        num_subfolders = (len(audio_files) + max_per_folder - 1) // max_per_folder
        print(f"  Creating {num_subfolders} subfolders for {len(audio_files)} files")

        # Create subfolders and distribute files
        subfolder_files = []
        for i in range(num_subfolders):
            start_idx = i * max_per_folder
            end_idx = min((i + 1) * max_per_folder, len(audio_files))
            subfolder_files.append(audio_files[start_idx:end_idx])

        # Create subfolders and move files
        new_metadata_entries = []
        for i, files_in_subfolder in enumerate(subfolder_files):
            subfolder_name = f"{i:03d}"
            subfolder_path = split_path / subfolder_name
            subfolder_path.mkdir(exist_ok=True)

            # Move files to subfolder
            for file_path in files_in_subfolder:
                new_path = subfolder_path / file_path.name
                shutil.move(str(file_path), str(new_path))

                # Update metadata entry with new path
                if original_metadata:
                    # Try different possible keys for filename
                    file_key = file_path.name
                    if file_key in original_metadata:
                        entry = original_metadata[file_key].copy()
                        # Update the file path in metadata
                        if "file_name" in entry:
                            entry["file_name"] = f"{file_path.name}"
                        elif "audio_file_path" in entry:
                            entry["audio_file_path"] = f"{file_path.name}"
                        new_metadata_entries.append(entry)

            # Create metadata.jsonl in each subfolder
            if new_metadata_entries:
                subfolder_metadata_path = subfolder_path / "metadata.jsonl"
                with open(subfolder_metadata_path, "w", encoding="utf-8") as f:
                    for entry in new_metadata_entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                new_metadata_entries = []

        # Remove original metadata.jsonl from root
        if metadata_files:
            metadata_files[0].unlink()

        print(f"  {split} split completed")


def main():
    for dir in [
        f"{DATA_DIR}/train",
        f"{DATA_DIR}/val",
        f"{DATA_DIR}/test",
    ]:
        os.makedirs(dir, exist_ok=True)

    vad = VAD()
    msd = MusicSilenceDetector()
    sil = Silence()

    match sys.argv[1]:
        case "download":
            # case "clean1":
            print("Downloading clean1")

            clean1 = load_dataset(
                "ammagra/english-arabic-speech-translation",
                split="test",
                streaming=True,
            )
            assert isinstance(clean1, IterableDataset)
            clean1 = (
                clean1.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .take(4000)
                .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            )
            process_dataset(clean1, vad, "speech", "speech_clean", 4000, "clean1")

            # case "clean2":
            print("Downloading clean2")

            clean2 = load_dataset(
                "MLCommons/peoples_speech",
                "clean",
                split="train",
                streaming=True,
            )
            assert isinstance(clean2, IterableDataset)
            clean2 = (
                clean2.select_columns("audio")
                .take(8000)
                .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            )
            process_dataset(clean2, vad, "speech", "speech_clean", 8000, "clean2")

            # case "clean3":
            print("Downloading clean3")

            clean3 = load_dataset(
                "MLCommons/peoples_speech",
                "clean",
                split="train",
                streaming=True,
            )
            assert isinstance(clean3, IterableDataset)
            clean3 = (
                clean3.select_columns("audio")
                .skip(8000)
                .take(8000)
                .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            )
            process_dataset(clean3, vad, "speech", "speech_clean", 8000, "clean3")

            # case "clean4":
            print("Downloading clean4")

            clean4 = load_dataset(
                "MLCommons/peoples_speech",
                "clean",
                split="train",
                streaming=True,
            )
            assert isinstance(clean4, IterableDataset)
            clean4 = (
                clean4.select_columns("audio")
                .skip(16000)
                .take(8000)
                .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            )
            process_dataset(clean4, vad, "speech", "speech_clean", 8000, "clean4")

            # case "clean5":
            # print("Downloading clean5")
            #
            # clean5 = load_dataset(
            #     "MLCommons/peoples_speech",
            #     "clean",
            #     split="train",
            #     streaming=True,
            # )
            # assert isinstance(clean5, IterableDataset)
            # clean5 = (
            #     clean5.select_columns("audio")
            #     .skip(24000)
            #     .take(8000)
            #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            # )
            # process_dataset(clean5, vad, "speech", "speech_clean", 8000, "clean5")

            # case "clean6":
            # print("Downloading clean6")
            #
            # clean6 = load_dataset(
            #     "MLCommons/peoples_speech",
            #     "clean",
            #     split="train",
            #     streaming=True,
            # )
            # assert isinstance(clean6, IterableDataset)
            # clean6 = (
            #     clean6.select_columns("audio")
            #     .skip(32000)
            #     .take(8000)
            #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            # )
            # process_dataset(clean6, vad, "speech", "speech_clean", 8000, "clean6")
            #
            # # case "clean7":
            # print("Downloading clean7")
            #
            # clean7 = load_dataset(
            #     "MLCommons/peoples_speech",
            #     "clean",
            #     split="train",
            #     streaming=True,
            # )
            # assert isinstance(clean7, IterableDataset)
            # clean7 = (
            #     clean7.select_columns("audio")
            #     .skip(40000)
            #     .take(8000)
            #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            # )
            # process_dataset(clean7, vad, "speech", "speech_clean", 8000, "clean7")
            #
            # # case "clean8":
            # print("Downloading clean8")
            #
            # clean8 = load_dataset(
            #     "MLCommons/peoples_speech",
            #     "clean",
            #     split="train",
            #     streaming=True,
            # )
            # assert isinstance(clean8, IterableDataset)
            # clean8 = (
            #     clean8.select_columns("audio")
            #     .skip(48000)
            #     .take(8000)
            #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            # )
            # process_dataset(clean8, vad, "speech", "speech_clean", 8000, "clean8")
            #
            # # case "clean9":
            # print("Downloading clean9")
            #
            # clean9 = load_dataset(
            #     "MLCommons/peoples_speech",
            #     "clean",
            #     split="train",
            #     streaming=True,
            # )
            # assert isinstance(clean9, IterableDataset)
            # clean9 = (
            #     clean9.select_columns("audio")
            #     .skip(56000)
            #     .take(8000)
            #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            # )
            # process_dataset(clean9, vad, "speech", "speech_clean", 8000, "clean9")

            # case "noisy":
            print("Downloading noisy")

            noisy = load_dataset(
                "Jzuluaga/atco2_corpus_1h", split="test", streaming=True
            )
            assert isinstance(noisy, IterableDataset)
            noisy = (
                noisy.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            )
            process_dataset(noisy, vad, "speech", "speech_noisy", 871, "noisy")

            # case "noisyenv":
            print("Downloading noisyenv")

            noisy_env = load_from_disk("noizeus_dataset")
            assert isinstance(noisy_env, Dataset)
            noisy_env = noisy_env.to_iterable_dataset()
            assert isinstance(noisy_env, IterableDataset)
            noisy_env = (
                noisy_env.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            )
            process_dataset(
                noisy_env, vad, "speech", "speech_noisyenv", 630, "noisyenv"
            )

            # case "jazz":
            print("Downloading jazz")

            jazz = load_dataset(
                "PlutoG99001/MusicGen-Jazz-Clean", split="train", streaming=True
            )
            assert isinstance(jazz, IterableDataset)
            jazz = (
                jazz.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                jazz, msd, "music", "music_acoustic", 50, "jazz", sil_thr=-20
            )

            # case "country":
            print("Downloading country")

            country = load_dataset(
                "ylacombe/music_genres_Country", split="train", streaming=True
            )
            assert isinstance(country, IterableDataset)
            country = (
                country.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                country, msd, "music", "music_acoustic", 142, "country", sil_thr=-25
            )

            # case "folk":
            print("Downloading folk")
            folk = load_dataset("lewtun/music_genres", split="train", streaming=True)
            assert isinstance(folk, IterableDataset)
            folk = (
                folk.shuffle(seed=RAND_SEED)
                .select_columns(["audio", "genre"])
                .filter(lambda row: row["genre"] == "Folk")
                .take(1000)
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                folk, msd, "music", "music_acoustic", 1000, "folk", sil_thr=-33
            )

            # case "pop":
            print("Downloading pop")

            pop = load_dataset(
                "memepottaboah/POPMUSIC1981", split="train", streaming=True
            )
            assert isinstance(pop, IterableDataset)
            pop = (
                pop.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                pop, msd, "music", "music_mainstream", 268, "pop", sil_thr=-22
            )

            # case "rock":
            print("Downloading rock")
            rock = load_dataset("lewtun/music_genres", split="train", streaming=True)
            assert isinstance(rock, IterableDataset)
            rock = (
                rock.shuffle(seed=RAND_SEED)
                .select_columns(["audio", "genre"])
                .filter(lambda row: row["genre"] == "Rock")
                .take(2000)
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                rock, msd, "music", "music_mainstream", 2000, "rock", sil_thr=-30
            )

            # case "electronic":
            print("Downloading electronic")

            electronic = load_dataset(
                "PlutoG99001/MusicGen-Electronic-Clean", split="train", streaming=True
            )
            assert isinstance(electronic, IterableDataset)
            electronic = (
                electronic.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                electronic,
                msd,
                "music",
                "music_electronic",
                52,
                "electronic",
                sil_thr=-25,
            )

            # case "instrumental":
            print("Downloading instrumental")
            instrumental = load_dataset(
                "PlutoG99001/MusicGen-Classical", split="train", streaming=True
            )
            assert isinstance(instrumental, IterableDataset)
            instrumental = (
                instrumental.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                instrumental,
                msd,
                "music",
                "music_instrumental",
                495,
                "instrumental",
                sil_thr=-27,
            )

            # case "vocal1":
            print("Downloading vocal1")

            vocal1 = load_dataset(
                "ccmusic-database/acapella", split="song1", streaming=True
            )
            assert isinstance(vocal1, IterableDataset)
            vocal1 = (
                vocal1.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                vocal1, msd, "music", "music_vocal", 22, "vocal1", sil_thr=-48
            )

            # case "vocal2":
            print("Downloading vocal2")

            vocal2 = load_dataset(
                "ccmusic-database/acapella", split="song2", streaming=True
            )
            assert isinstance(vocal2, IterableDataset)
            vocal2 = (
                vocal2.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                vocal2, msd, "music", "music_vocal", 22, "vocal2", sil_thr=-48
            )

            # case "vocal3":
            print("Downloading vocal3")

            vocal3 = load_dataset(
                "ccmusic-database/acapella", split="song3", streaming=True
            )
            assert isinstance(vocal3, IterableDataset)
            vocal3 = (
                vocal3.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column(
                    "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
                )
            )
            process_dataset(
                vocal3, msd, "music", "music_vocal", 22, "vocal3", sil_thr=-48
            )

            # case "noise":
            print("Downloading noise")

            noise = load_from_disk("musan_noise_dataset")
            assert isinstance(noise, Dataset)
            noise = noise.to_iterable_dataset()
            assert isinstance(noise, IterableDataset)
            noise = (
                noise.shuffle(seed=RAND_SEED)
                .select_columns("audio")
                .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
            )
            process_dataset(noise, sil, "noise", "noise", 169, "noise")

        case "combine":
            combine_metas()

        case "split":
            split_files()


if __name__ == "__main__":
    main()
    time.sleep(1)
