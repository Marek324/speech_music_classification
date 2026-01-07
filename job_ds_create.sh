#!/bin/bash
#PBS -N DatasetCreateJob
#PBS -l walltime=15:00:00
#PBS -l select=1:ncpus=32:mem=200gb:scratch_local=250gb

PROJ_DIR=/storage/brno2/home/marek324/bp

source $PROJ_DIR/setup.sh

# logs
LOGS="$PROJ_DIR/logs/ds_create/$PBS_JOBID"
mkdir -p "$LOGS"

cd /storage/brno2/home/marek324/dataset

export HF_HUB_DOWNLOAD_TIMEOUT=20
export HF_HUB_ETAG_TIMEOUT=20
export HF_HUB_TIMEOUT=20

#uv run --active $PROJ_DIR/dataset_scripts/noise.py download >  $LOGS/out_noise_download.log 2>  $LOGS/err_noise_download.log 
#uv run --active $PROJ_DIR/dataset_scripts/noise.py dataset >  $LOGS/out_noise_dataset.log 2>  $LOGS/err_noise_dataset.log 

#uv run --active $PROJ_DIR/dataset_scripts/noizeus.py download >  $LOGS/out_noizeus_download.log 2>  $LOGS/err_noizeus_download.log 
#uv run --active $PROJ_DIR/dataset_scripts/noizeus.py dataset >  $LOGS/out_noizeus_dataset.log 2>  $LOGS/err_noizeus_dataset.log 

#uv run --active $PROJ_DIR/dataset_scripts/process.py download >  $LOGS/out_process_download.log 2>  $LOGS/err_process_download.log 
#uv run --active $PROJ_DIR/dataset_scripts/process.py combine >  $LOGS/out_process_combine.log 2>  $LOGS/err_process_combine.log 
#uv run --active $PROJ_DIR/dataset_scripts/process.py split >  $LOGS/out_process_split.log 2>  $LOGS/err_process_split.log 

#uv run --active $PROJ_DIR/dataset_scripts/process.py upload >  $LOGS/out_process_upload.log 2>  $LOGS/err_process_upload.log 
uv run --active $PROJ_DIR/dataset_scripts/process.py  >  $LOGS/out.log 2>  $LOGS/err.log 
