#!/bin/bash

HOME_DIR=/storage/brno2/home/marek324/bp
cd $HOME_DIR
source .venv/bin/activate
module load ffmpeg

uv run src/main.py > $HOME_DIR/out.log 2> $HOME_DIR/err.log
