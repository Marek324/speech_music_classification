#!/bin/bash
#PBS -N DatasetStatsJob
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=32:mem=150gb:scratch_local=250gb

PROJ_DIR=/storage/brno2/home/marek324/bp

source $PROJ_DIR/setup.sh

# logs
LOGS="$PROJ_DIR/logs/ds_stats/$PBS_JOBID"
mkdir -p "$LOGS"

cd $PROJ_DIR

uv run --active $PROJ_DIR/src/main.py dataset_stats --config-file "$PROJ_DIR/config.yaml" > "$LOGS/out.log" 2> "$LOGS/err.log"
