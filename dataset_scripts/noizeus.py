import os
import shutil
import sys
import zipfile
from itertools import product
from pathlib import Path

import urllib3
import requests
from tqdm import tqdm
from datasets import IterableDataset, Dataset, Audio, load_from_disk


def download():
    os.makedirs("noizeus/zips", exist_ok=True)
    base = "https://ecs.utdallas.edu/loizou/speech/noizeus/"
    noises = [
        "train",
        "babble",
        "exhibition",
        "restaurant",
        "street",
        "airport",
        "station",
    ]
    snrs = ["_0dB", "_5dB", "_10dB"]

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    for noise, snr in tqdm(product(noises, snrs)):
        name = f"{noise}{snr}"
        url = f"{base}{name}.zip"
        res = requests.get(url, verify=False)
        with open(f"noizeus/zips/{name}.zip", "wb") as f:
            f.write(res.content)

        with zipfile.ZipFile(f"noizeus/zips/{name}.zip", "r") as zip_ref:
            zip_ref.extractall(f"noizeus/{name}")

        os.remove(f"noizeus/zips/{name}.zip")

        nested_dir = f"noizeus/{name}/{snr.replace('_', '')}"
        if os.path.exists(nested_dir):
            for filename in os.listdir(nested_dir):
                shutil.move(
                    os.path.join(nested_dir, filename), f"noizeus/{name}/{filename}"
                )
            os.rmdir(nested_dir)

    os.rmdir("noizeus/zips")


def create_dataset():
    files = []
    for dir in Path("noizeus").iterdir():
        if dir.is_dir():
            for wav in dir.glob("*.wav"):
                files.append(str(wav))

    dataset = Dataset.from_dict({"audio": files})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))

    dataset.save_to_disk("noizeus_dataset")


def test():
    dataset = load_from_disk("noizeus_dataset")
    assert isinstance(dataset, Dataset)
    dataset = dataset.to_iterable_dataset()
    assert isinstance(dataset, IterableDataset)

    count = 0
    for row in dataset:
        count += 1

    print(count)


if __name__ == "__main__":
    arg = sys.argv[1]
    match arg:
        case "download":
            download()
        case "dataset":
            create_dataset()
        case "test":
            test()
