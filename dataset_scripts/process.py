import io
import os
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
MAX_FILES_PER_FOLDER = 9000
SR = 16000


def create_label(label_type: str, start: int, end: int):
    """
    Creates a label dictionary with all keys present (speech, music, inactive).
    Unused keys are set to None. This prevents schema mismatch errors in HF datasets.
    """
    segment = {"start": start, "end": end}
    
    return {
        "speech": segment if label_type == "speech" else None,
        "music": segment if label_type == "music" else None,
        "inactive": segment if label_type == "inactive" else None,
    }

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

        return [create_label("inactive", 0, siglen)]


class VAD:
    def __init__(self):
        self.model = load_silero_vad()

    def label(
        self, decoder: AudioDecoder | dict[str, str | bytes], _
    ) -> Optional[list[dict[str, dict[str, int]]]]:
        def stoms(sample: int) -> int:
            return int(sample * 1000 / SR)

        assert isinstance(decoder, AudioDecoder)

        try:
            x = decoder.get_all_samples()
        except RuntimeError:
            return None

        stamps: list[dict[str, int]] = get_speech_timestamps(x.data, self.model)

        if not stamps:
            return [create_label("inactive", 0, len(x.data))]

        labels: list[dict[str, dict[str, int]]] = []
        prev_end = 0

        for stamp in stamps:
            if stamp["start"] > prev_end:
                labels.append(
                    {
                        create_label("inactive", stoms(prev_end), stoms(stamp["start"]))
                    }
                )

            labels.append(create_label("speech", stoms(stamp["start"]), stoms(stamp["end"])))
            prev_end = stamp["end"]

        if prev_end < len(x.data):
            labels.append(
                create_label("inactive", stoms(prev_end), stoms(len(x.data)))
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
        )

        if not sil_stamps:
            return [create_label("music", 0, len(seg))]

        prev_end = 0

        for stamp in sil_stamps:
            if stamp[0] > prev_end:
                labels.append(
                    create_label("music", prev_end, stamp[0])
                )

            labels.append(create_label("inactive", stamp[0], stamp[1]))
            prev_end = stamp[1]

        if prev_end < len(seg):
            labels.append(create_label("music", prev_end, len(seg)))

        return labels


class DirectWriter:
    """
    Writes files directly to data/{split}/{folder_idx}/
    Handles folder rotation automatically.
    """
    def __init__(self, split: str):
        self.split = split
        self.base_dir = Path(DATA_DIR) / split
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._init_state()

    def _init_state(self):
        # Find existing numeric subfolders
        folders = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not folders:
            self.current_folder_idx = 0
            self.current_folder_path = self.base_dir / "000"
            self.current_folder_path.mkdir(exist_ok=True)
            self.current_file_count = 0
        else:
            # Sort to find the last one
            folders.sort(key=lambda x: int(x.name))
            self.current_folder_idx = int(folders[-1].name)
            self.current_folder_path = folders[-1]
            # Count existing wav files to resume correctly
            self.current_file_count = len(list(self.current_folder_path.glob("*.wav")))

    def write(
        self,
        audio: AudioDecoder | dict[str, str | bytes],
        labels: list[dict[str, dict[str, int]]],
        name: str,
        idx: int,
        _class: str,
        subclass: str,
    ):
        # Rotate folder if full
        if self.current_file_count >= MAX_FILES_PER_FOLDER:
            self.current_folder_idx += 1
            self.current_folder_path = self.base_dir / f"{self.current_folder_idx:03d}"
            self.current_folder_path.mkdir(exist_ok=True)
            self.current_file_count = 0

        file_name = f"{name}_{idx:05d}.wav"
        file_path = self.current_folder_path / file_name

        # Avoid re-processing if file exists (optional safety)
        if file_path.exists():
            return

        # 1. Save Audio
        if isinstance(audio, AudioDecoder):
            audio_data = audio.get_all_samples().data
            if hasattr(audio_data, "cpu"):
                audio_data = audio_data.cpu()
            audio_data = audio_data.numpy().squeeze()
            sf.write(file_path, audio_data, samplerate=SR)
        else:
            if audio["bytes"] is not None:
                seg = AudioSegment.from_file(io.BytesIO(audio["bytes"]))
            else:
                seg = AudioSegment.from_file(audio["path"])
            seg.export(file_path, format="wav")

        # 2. Append Metadata
        entry = {
            "file_name": file_name,
            "class": _class,
            "subclass": subclass,
            "labels": labels,
        }
        
        # Open in append mode per item to ensure safety on crash
        with jsonlines.open(self.current_folder_path / "metadata.jsonl", mode="a") as writer:
            writer.write(entry)

        self.current_file_count += 1


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
    
    # We define splits but create Writers just-in-time
    splits = [
        (dataset.take(train_end), "train"),
        (dataset.skip(train_end).take(val_end - train_end), "val"),
        (dataset.skip(val_end), "test"),
    ]

    for split_data, split_name in splits:
        # Initialize writer (scans disk to resume where previous dataset left off)
        writer = DirectWriter(split_name)
        
        # Local index for this specific dataset (clean1_00001, clean1_00002...)
        local_idx = 0 
        
        for row in tqdm(split_data, desc=f"{name} - {split_name}"):
            audio = row["audio"]
            labels = silence_detector.label(audio, sil_thr)
            
            if labels is None:
                print("Invalid data, skipping", file=sys.stderr)
                continue
                
            writer.write(audio, labels, name, local_idx, _class, subclass)
            local_idx += 1

    print(f"Finished {name}\n")



def main():
    vad = VAD()
    msd = MusicSilenceDetector()
    sil = Silence()

    # --- SPEECH ---
    print("Downloading clean1")
    clean1 = load_dataset("ammagra/english-arabic-speech-translation", split="test", streaming=True)
    clean1 = clean1.shuffle(seed=RAND_SEED).select_columns("audio").take(4000).cast_column("audio", Audio(sampling_rate=SR, num_channels=1))
    process_dataset(clean1, vad, "speech", "speech_clean", 4000, "clean1")

    print("Downloading clean2")
    clean2 = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
    clean2 = clean2.select_columns("audio").take(8000).cast_column("audio", Audio(sampling_rate=SR, num_channels=1))
    process_dataset(clean2, vad, "speech", "speech_clean", 8000, "clean2")

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

    # --- MUSIC ---

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



if __name__ == "__main__":
    main()
    time.sleep(1)

# def main():
#     for dir in [
#         f"{DATA_DIR}/train",
#         f"{DATA_DIR}/val",
#         f"{DATA_DIR}/test",
#     ]:
#         os.makedirs(dir, exist_ok=True)

#     vad = VAD()
#     msd = MusicSilenceDetector()
#     sil = Silence()

#     match sys.argv[1]:
#         case "download":
#             # case "clean1":
#             print("Downloading clean1")

#             clean1 = load_dataset(
#                 "ammagra/english-arabic-speech-translation",
#                 split="test",
#                 streaming=True,
#             )
#             assert isinstance(clean1, IterableDataset)
#             clean1 = (
#                 clean1.shuffle(seed=RAND_SEED)
#                 .select_columns("audio")
#                 .take(4000)
#                 .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             )
#             process_dataset(clean1, vad, "speech", "speech_clean", 4000, "clean1")

#             # case "clean2":
#             print("Downloading clean2")

#             clean2 = load_dataset(
#                 "MLCommons/peoples_speech",
#                 "clean",
#                 split="train",
#                 streaming=True,
#             )
#             assert isinstance(clean2, IterableDataset)
#             clean2 = (
#                 clean2.select_columns("audio")
#                 .take(8000)
#                 .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             )
#             process_dataset(clean2, vad, "speech", "speech_clean", 8000, "clean2")

#             # case "clean3":
#             #print("Downloading clean3")

#             #clean3 = load_dataset(
#             #    "MLCommons/peoples_speech",
#             #    "clean",
#             #    split="train",
#             #    streaming=True,
#             #)
#             #assert isinstance(clean3, IterableDataset)
#             #clean3 = (
#             #    clean3.select_columns("audio")
#             #    .skip(8000)
#             #    .take(8000)
#             #    .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             #)
#             #process_dataset(clean3, vad, "speech", "speech_clean", 8000, "clean3")

#             # case "clean4":
#             # print("Downloading clean4")
#             #
#             # clean4 = load_dataset(
#             #     "MLCommons/peoples_speech",
#             #     "clean",
#             #     split="train",
#             #     streaming=True,
#             # )
#             # assert isinstance(clean4, IterableDataset)
#             # clean4 = (
#             #     clean4.select_columns("audio")
#             #     .skip(16000)
#             #     .take(8000)
#             #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             # )
#             # process_dataset(clean4, vad, "speech", "speech_clean", 8000, "clean4")
#             #
#             # case "clean5":
#             # print("Downloading clean5")
#             #
#             # clean5 = load_dataset(
#             #     "MLCommons/peoples_speech",
#             #     "clean",
#             #     split="train",
#             #     streaming=True,
#             # )
#             # assert isinstance(clean5, IterableDataset)
#             # clean5 = (
#             #     clean5.select_columns("audio")
#             #     .skip(24000)
#             #     .take(8000)
#             #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             # )
#             # process_dataset(clean5, vad, "speech", "speech_clean", 8000, "clean5")

#             # case "clean6":
#             # print("Downloading clean6")
#             #
#             # clean6 = load_dataset(
#             #     "MLCommons/peoples_speech",
#             #     "clean",
#             #     split="train",
#             #     streaming=True,
#             # )
#             # assert isinstance(clean6, IterableDataset)
#             # clean6 = (
#             #     clean6.select_columns("audio")
#             #     .skip(32000)
#             #     .take(8000)
#             #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             # )
#             # process_dataset(clean6, vad, "speech", "speech_clean", 8000, "clean6")
#             #
#             # # case "clean7":
#             # print("Downloading clean7")
#             #
#             # clean7 = load_dataset(
#             #     "MLCommons/peoples_speech",
#             #     "clean",
#             #     split="train",
#             #     streaming=True,
#             # )
#             # assert isinstance(clean7, IterableDataset)
#             # clean7 = (
#             #     clean7.select_columns("audio")
#             #     .skip(40000)
#             #     .take(8000)
#             #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             # )
#             # process_dataset(clean7, vad, "speech", "speech_clean", 8000, "clean7")
#             #
#             # # case "clean8":
#             # print("Downloading clean8")
#             #
#             # clean8 = load_dataset(
#             #     "MLCommons/peoples_speech",
#             #     "clean",
#             #     split="train",
#             #     streaming=True,
#             # )
#             # assert isinstance(clean8, IterableDataset)
#             # clean8 = (
#             #     clean8.select_columns("audio")
#             #     .skip(48000)
#             #     .take(8000)
#             #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             # )
#             # process_dataset(clean8, vad, "speech", "speech_clean", 8000, "clean8")
#             #
#             # # case "clean9":
#             # print("Downloading clean9")
#             #
#             # clean9 = load_dataset(
#             #     "MLCommons/peoples_speech",
#             #     "clean",
#             #     split="train",
#             #     streaming=True,
#             # )
#             # assert isinstance(clean9, IterableDataset)
#             # clean9 = (
#             #     clean9.select_columns("audio")
#             #     .skip(56000)
#             #     .take(8000)
#             #     .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             # )
#             # process_dataset(clean9, vad, "speech", "speech_clean", 8000, "clean9")

#             # case "noisy":
#             #print("Downloading noisy")

#             #noisy = load_dataset(
#             #    "Jzuluaga/atco2_corpus_1h", split="test", streaming=True
#             #)
#             #assert isinstance(noisy, IterableDataset)
#             #noisy = (
#             #    noisy.shuffle(seed=RAND_SEED)
#             #    .select_columns("audio")
#             #    .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             #)
#             #process_dataset(noisy, vad, "speech", "speech_noisy", 871, "noisy")

#             ## case "noisyenv":
#             #print("Downloading noisyenv")

#             #noisy_env = load_from_disk("noizeus_dataset")
#             #assert isinstance(noisy_env, Dataset)
#             #noisy_env = noisy_env.to_iterable_dataset()
#             #assert isinstance(noisy_env, IterableDataset)
#             #noisy_env = (
#             #    noisy_env.shuffle(seed=RAND_SEED)
#             #    .select_columns("audio")
#             #    .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             #)
#             #process_dataset(
#             #    noisy_env, vad, "speech", "speech_noisyenv", 630, "noisyenv"
#             #)

#             # case "jazz":
#             print("Downloading jazz")

#             jazz = load_dataset(
#                 "PlutoG99001/MusicGen-Jazz-Clean", split="train", streaming=True
#             )
#             assert isinstance(jazz, IterableDataset)
#             jazz = (
#                 jazz.shuffle(seed=RAND_SEED)
#                 .select_columns("audio")
#                 .cast_column(
#                     "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#                 )
#             )
#             process_dataset(
#                 jazz, msd, "music", "music_acoustic", 50, "jazz", sil_thr=-20
#             )

#             # case "country":
#             print("Downloading country")

#             country = load_dataset(
#                 "ylacombe/music_genres_Country", split="train", streaming=True
#             )
#             assert isinstance(country, IterableDataset)
#             country = (
#                 country.shuffle(seed=RAND_SEED)
#                 .select_columns("audio")
#                 .cast_column(
#                     "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#                 )
#             )
#             process_dataset(
#                 country, msd, "music", "music_acoustic", 142, "country", sil_thr=-25
#             )

#             # case "folk":
#             #print("Downloading folk")
#             #folk = load_dataset("lewtun/music_genres", split="train", streaming=True)
#             #assert isinstance(folk, IterableDataset)
#             #folk = (
#             #    folk.shuffle(seed=RAND_SEED)
#             #    .select_columns(["audio", "genre"])
#             #    .filter(lambda row: row["genre"] == "Folk")
#             #    .take(1000)
#             #    .cast_column(
#             #        "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#             #    )
#             #)
#             #process_dataset(
#             #    folk, msd, "music", "music_acoustic", 1000, "folk", sil_thr=-33
#             #)

#             ## case "pop":
#             #print("Downloading pop")

#             #pop = load_dataset(
#             #    "memepottaboah/POPMUSIC1981", split="train", streaming=True
#             #)
#             #assert isinstance(pop, IterableDataset)
#             #pop = (
#             #    pop.shuffle(seed=RAND_SEED)
#             #    .select_columns("audio")
#             #    .cast_column(
#             #        "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#             #    )
#             #)
#             #process_dataset(
#             #    pop, msd, "music", "music_mainstream", 268, "pop", sil_thr=-22
#             #)

#             ## case "rock":
#             #print("Downloading rock")
#             #rock = load_dataset("lewtun/music_genres", split="train", streaming=True)
#             #assert isinstance(rock, IterableDataset)
#             #rock = (
#             #    rock.shuffle(seed=RAND_SEED)
#             #    .select_columns(["audio", "genre"])
#             #    .filter(lambda row: row["genre"] == "Rock")
#             #    .take(2000)
#             #    .cast_column(
#             #        "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#             #    )
#             #)
#             #process_dataset(
#             #    rock, msd, "music", "music_mainstream", 2000, "rock", sil_thr=-30
#             #)

#             # case "electronic":
#             print("Downloading electronic")

#             electronic = load_dataset(
#                 "PlutoG99001/MusicGen-Electronic-Clean", split="train", streaming=True
#             )
#             assert isinstance(electronic, IterableDataset)
#             electronic = (
#                 electronic.shuffle(seed=RAND_SEED)
#                 .select_columns("audio")
#                 .cast_column(
#                     "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#                 )
#             )
#             process_dataset(
#                 electronic,
#                 msd,
#                 "music",
#                 "music_electronic",
#                 52,
#                 "electronic",
#                 sil_thr=-25,
#             )

#             # case "instrumental":
#             print("Downloading instrumental")
#             instrumental = load_dataset(
#                 "PlutoG99001/MusicGen-Classical", split="train", streaming=True
#             )
#             assert isinstance(instrumental, IterableDataset)
#             instrumental = (
#                 instrumental.shuffle(seed=RAND_SEED)
#                 .select_columns("audio")
#                 .cast_column(
#                     "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#                 )
#             )
#             process_dataset(
#                 instrumental,
#                 msd,
#                 "music",
#                 "music_instrumental",
#                 495,
#                 "instrumental",
#                 sil_thr=-27,
#             )

#             # case "vocal1":
#             #print("Downloading vocal1")

#             #vocal1 = load_dataset(
#             #    "ccmusic-database/acapella", split="song1", streaming=True
#             #)
#             #assert isinstance(vocal1, IterableDataset)
#             #vocal1 = (
#             #    vocal1.shuffle(seed=RAND_SEED)
#             #    .select_columns("audio")
#             #    .cast_column(
#             #        "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#             #    )
#             #)
#             #process_dataset(
#             #    vocal1, msd, "music", "music_vocal", 22, "vocal1", sil_thr=-48
#             #)

#             ## case "vocal2":
#             #print("Downloading vocal2")

#             #vocal2 = load_dataset(
#             #    "ccmusic-database/acapella", split="song2", streaming=True
#             #)
#             #assert isinstance(vocal2, IterableDataset)
#             #vocal2 = (
#             #    vocal2.shuffle(seed=RAND_SEED)
#             #    .select_columns("audio")
#             #    .cast_column(
#             #        "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#             #    )
#             #)
#             #process_dataset(
#             #    vocal2, msd, "music", "music_vocal", 22, "vocal2", sil_thr=-48
#             #)

#             ## case "vocal3":
#             #print("Downloading vocal3")

#             #vocal3 = load_dataset(
#             #    "ccmusic-database/acapella", split="song3", streaming=True
#             #)
#             #assert isinstance(vocal3, IterableDataset)
#             #vocal3 = (
#             #    vocal3.shuffle(seed=RAND_SEED)
#             #    .select_columns("audio")
#             #    .cast_column(
#             #        "audio", Audio(sampling_rate=sr, num_channels=1, decode=False)
#             #    )
#             #)
#             #process_dataset(
#             #    vocal3, msd, "music", "music_vocal", 22, "vocal3", sil_thr=-48
#             #)

#             ## case "noise":
#             #print("Downloading noise")

#             #noise = load_from_disk("musan_noise_dataset")
#             #assert isinstance(noise, Dataset)
#             #noise = noise.to_iterable_dataset()
#             #assert isinstance(noise, IterableDataset)
#             #noise = (
#             #    noise.shuffle(seed=RAND_SEED)
#             #    .select_columns("audio")
#             #    .cast_column("audio", Audio(sampling_rate=sr, num_channels=1))
#             #)
#             #process_dataset(noise, sil, "noise", "noise", 169, "noise")

#         case "combine":
#             combine_metas()

#         case "split":
#             split_files()

#         case "upload":
#             upload()


# if __name__ == "__main__":
#     main()
#     time.sleep(1)
