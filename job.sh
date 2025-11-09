#!/bin/bash

cd /storage/brno2/home/marek324/bp
source .venv/bin/activate

uv run src/main.py > out.log 2> err.log
