#!/bin/bash
#PBS -N DecisionTreeEval
#PBS -l walltime=15:00:00
#PBS -l select=1:ncpus=32:mem=150gb:scratch_local=150gb

PROJ_DIR=/storage/brno2/home/marek324/bp

source $PROJ_DIR/setup.sh

# logs
LOGS="$PROJ_DIR/logs/dt_eval/$PBS_JOBID"
mkdir -p "$LOGS"

cd $PROJ_DIR

uv run --active $PROJ_DIR/src/main.py train --config-file "$PROJ_DIR/config.yaml" > $LOGS/out_train.log 2> $LOGS/err_train.log
uv run --active $PROJ_DIR/src/main.py eval --config-file "$PROJ_DIR/config.yaml" > $LOGS/out_eval.log 2> $LOGS/err_eval.log
