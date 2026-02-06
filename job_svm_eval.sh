#!/bin/bash
#PBS -N SVMEval
#PBS -l walltime=11:00:00
#PBS -l select=1:ncpus=64:mem=450gb:scratch_local=400gb

PROJ_DIR=/storage/brno2/home/marek324/bp

export SCRATCH="$SCRATCHDIR" 
export HF_HOME="$SCRATCHDIR/hf_home"
mkdir -p $HF_HOME

source $PROJ_DIR/setup.sh

# logs
LOGS="$PROJ_DIR/logs/svm_eval/$PBS_JOBID"
mkdir -p "$LOGS"

cd $PROJ_DIR

echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_XET_CACHE: $HF_XET_CACHE"
echo "HF_ASSETS_CACHE: $HF_ASSETS_CACHE"


uv run --active $PROJ_DIR/src/main.py train svm --config-file "$PROJ_DIR/config.yaml" > $LOGS/out_train.log 2> $LOGS/err_train.log
uv run --active $PROJ_DIR/src/main.py eval svm --config-file "$PROJ_DIR/config.yaml" > $LOGS/out_eval.log 2> $LOGS/err_eval.log

quota -s
du -sh $HF_HOME
