#!/bin/bash
#PBS -N DatasetCreateJob
#PBS -l walltime=15:00:00
#PBS -l select=1:ncpus=32:mem=150gb:scratch_local=150gb

PROJ_DIR=/storage/brno2/home/marek324/bp

source $PROJ_DIR/setup.sh

# logs
LOGS="$PROJ_DIR/logs/ds_create/$PBS_JOBID"
mkdir -p "$LOGS"

uv run $PROJ_DIR/dataset_scripts/noise.py download >  $LOGS/out_noise_download.log 2>  $LOGS/err_noise_download.log 
uv run $PROJ_DIR/dataset_scripts/noise.py dataset >  $LOGS/out_noise_dataset.log 2>  $LOGS/err_noise_dataset.log 

uv run $PROJ_DIR/dataset_scripts/noizeus.py download >  $LOGS/out_noizeus_download.log 2>  $LOGS/err_noizeus_download.log 
uv run $PROJ_DIR/dataset_scripts/noizeus.py dataset >  $LOGS/out_noizeus_dataset.log 2>  $LOGS/err_noizeus_dataset.log 

uv run $PROJ_DIR/dataset_scripts/process.py download >  $LOGS/out_process_download.log 2>  $LOGS/err_process_download.log 
uv run $PROJ_DIR/dataset_scripts/process.py combine >  $LOGS/out_process_combine.log 2>  $LOGS/err_process_sombine.log 
uv run $PROJ_DIR/dataset_scripts/process.py split >  $LOGS/out_process_split.log 2>  $LOGS/err_process_split.log 
