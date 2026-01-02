#!/bin/bash
#PBS -N DatasetUploadJob
#PBS -l walltime=15:00:00
#PBS -l select=1:ncpus=32:mem=150gb:scratch_local=150gb

BASE_DIR=/storage/brno2/home/marek324
PROJ_DIR="$BASE_DIR/bp"

source $PROJ_DIR/setup.sh

# logs
LOGS="$PROJ_DIR/logs/ds_upload/$PBS_JOBID"
mkdir -p "$LOGS"

uvx hf upload-large-folder --repo-type dataset --no-bars Marek324/speech-music-classification $BASE_DIR/dataset
