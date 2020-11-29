#!/bin/bash

export DATA_ROOT=/ssd/data

export N_GPU=`nvidia-smi -L | wc -l`
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/pretrained-cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

pip install --user --editable ./
mkdir -p $DATA_ROOT/outputs
