#!/bin/bash
set -euo pipefail

###########################
#       DIRECTORIES       #
###########################

BASE_PATH="/storage/brno2/home/marek324"


PROJ_PATH="$BASE_PATH/bp"
REQ_FILE="$PROJ_PATH/requirements.txt"
ENV_FILE="$PROJ_PATH/.env"

VENV_BASE="$BASE_PATH/env"
mkdir -p "$VENV_BASE"

PROJ_NAME="bp"
VENV_PATH="$VENV_BASE/$PROJ_NAME"

CACHE_BASE="${SCRATCH}/cache"
TMP_BASE="${SCRATCH}/tmp"
mkdir -p "$CACHE_BASE/uv" "$TMP_BASE"

export TMP="$TMP_BASE"
export TMPDIR="$TMP_BASE"
export UV_CACHE_DIR="$CACHE_BASE/uv"

###########################
#         MODULES         #
###########################

module purge
module load ffmpeg
module load python

###########################
#      BOOTSTRAP UV       #
###########################

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "[INFO] 'uv' not found. Installing via pip..."
    python -m pip install --user uv
fi

###########################
#          VENV           #
###########################

if [ ! -d "$VENV_PATH" ]; then
    echo "[INFO] Creating virtual environment with Python 3.13 at $VENV_PATH"
    uv venv "$VENV_PATH" --python 3.13
fi

source "$VENV_PATH/bin/activate"

echo "[INFO] Syncing requirements..."
uv pip install -r "$REQ_FILE"

echo "[INFO] Virtual environment ready at $VENV_PATH"


###########################
#       HUGGINGFACE       #
###########################

source $ENV_FILE # HF_TOKEN

uvx hf auth login --token $HF_TOKEN

DATASET_DIR="$BASE_PATH/dataset_$PROJ_NAME"
export HF_HUB_CACHE="$DATASET_DIR"

export DATASET_NAME=Marek324/speech-music-classification

#uvx hf download $DATASET_NAME --repo-type dataset

set +euo pipefail
