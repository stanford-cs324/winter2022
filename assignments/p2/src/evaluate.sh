#!/bin/bash
MODEL_DIR=$1
EVAL_DATA_DIR=$2

python src/evaluate_wordlength_model.py \
    --model_dir ${MODEL_DIR} \
    --eval_data_dir ${EVAL_DATA_DIR}
