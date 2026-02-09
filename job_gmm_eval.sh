#!/bin/bash
#PBS -N GMMEval
#PBS -l walltime=11:00:00
#PBS -l select=1:ncpus=64:mem=450gb:scratch_local=400gb

PROJ_DIR=/storage/brno2/home/marek324/bp

export SCRATCH="$SCRATCHDIR" 
export HF_HOME="$SCRATCHDIR/hf_home"
mkdir -p $HF_HOME

source $PROJ_DIR/setup.sh

# logs
LOGS="$PROJ_DIR/logs/gmm_eval/$PBS_JOBID"
mkdir -p "$LOGS"

cd $PROJ_DIR

uv run --active $PROJ_DIR/src/main.py train  --config-file "$PROJ_DIR/config_gmm.toml" > $LOGS/out_train.log 2> $LOGS/err_train.log
uv run --active $PROJ_DIR/src/main.py eval --config-file "$PROJ_DIR/config_gmm.toml" > $LOGS/out_eval.log 2> $LOGS/err_eval.log
