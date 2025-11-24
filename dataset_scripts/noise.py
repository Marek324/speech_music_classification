import os
import sys
from pathlib import Path

import boto3
from datasets import Audio, Dataset, IterableDataset, load_from_disk
from dotenv import load_dotenv


def download():
    load_dotenv()

    ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        region_name="auto",
    )

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket="musan-noise")

    file_count = 0

    os.makedirs("musan_noise", exist_ok=True)
    for page in pages:
        if "Contents" not in page:
            return

        for obj in page["Contents"]:
            key = obj["Key"]

            if key.endswith("/"):
                continue

            local_file_path = os.path.join("musan_noise", key)

            s3_client.download_file("musan-noise", key, local_file_path)
            file_count += 1

    print(f"File count: {file_count}")


def create_dataset():
    files = []
    for wav in Path("musan_noise").glob("*.wav"):
        files.append(str(wav))

    dataset = Dataset.from_dict({"audio": files})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))

    dataset.save_to_disk("musan_noise_dataset")


def test():
    dataset = load_from_disk("musan_noise_dataset")
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
