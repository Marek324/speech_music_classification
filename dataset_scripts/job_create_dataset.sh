#!/bin/sh

HOME_DIR=/storage/brno2/home/marek324/bp
DS_DIR=$HOME_DIR/dataset_scripts
LOGS_DIR=$DS_DIR/job_logs


source $HOME_DIR/.venv/bin/activate
module load ffmpeg

source $HOME_DIR/.env
source $DS_DIR/.env

mkdir -p $LOGS_DIR

uv run hf auth login --token $HF_TOKEN --add-to-git-credential

uv run $DS_DIR/noise.py download > $LOGS_DIR/out_noise_download.log 2> $LOGS_DIR/err_noise_download.log
uv run $DS_DIR/noise.py dataset > $LOGS_DIR/out_noise_dataset.log 2> $LOGS_DIR/err_noise_dataset.log

uv run $DS_DIR/noizeus.py download > $LOGS_DIR/out_noizeus_download.log 2> $LOGS_DIR/err_noizeus_download.log
uv run $DS_DIR/noizeus.py dataset > $LOGS_DIR/out_noizeus_dataset.log 2> $LOGS_DIR/err_noizeus_dataset.log

uv run $DS_DIR/process.py download > $LOGS_DIR/out_process_download.log 2> $LOGS_DIR/err_process_download.log
uv run $DS_DIR/process.py combine > $LOGS_DIR/out_process_combine.log 2> $LOGS_DIR/err_process_combine.log
uv run $DS_DIR/process.py split > $LOGS_DIR/out_process_split.log 2> $LOGS_DIR/err_process_split.log

