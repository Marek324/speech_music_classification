#!/bin/bash

HOME_DIR=/storage/brno2/home/marek324/bp
cd $HOME_DIR
source .venv/bin/activate
module load ffmpeg

source .env

uv run huggingface-cli download Marek324/speech-music-classification --repo-type dataset --token $HF_TOKEN > out_hf_cli_download.log 2> err_hf_cli_download.log

#uv run src/main.py train > out_train.log 2> err_train.log
uv run src/main.py eval > out_eval.log 2> err_eval.log
