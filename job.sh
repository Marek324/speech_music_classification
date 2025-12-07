#!/bin/bash

HOME_DIR=/storage/brno2/home/marek324/bp
cd $HOME_DIR
source .venv/bin/activate
module load ffmpeg

uv run src/main.py "$@" > out.log 2> err.log
