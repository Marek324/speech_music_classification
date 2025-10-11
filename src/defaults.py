# defaults.py
# Marek Hric

from pathlib import Path

SAMPLE_RATE = 44100  # Default sample rate
FRAME_LEN_MS = 20  # Frame length in milliseconds
FRAME_HOP_MS = 10  # Frame hop in milliseconds
SEGMENT_LEN_MS = 300  # Segment length in milliseconds

SRC_PATH = f"{Path(__file__).parent.parent}"
DATASET_PATH = f"{SRC_PATH}/dataset"  # Path to the dataset
REF_PATH = f"{SRC_PATH}/dataset/reference.csv"  # Path to the reference file
