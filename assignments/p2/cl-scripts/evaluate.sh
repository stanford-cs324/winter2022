#!/bin/bash

set -x

WORKSHEET_NAME=cs324-project2 # TODO change this!!

# first, run on the vanilla gpt2
RUN_NAME="eval_openwebtext_wordlength"
EVAL_DATA_DIR=src/wordlength_eval_data
SEED=1111 # random seed
MODEL_DIR=gpt2 # without any further training
CMD="python src/evaluate_wordlength_model.py --model_dir ${MODEL_DIR} --eval_data_dir ${EVAL_DATA_DIR}"
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 32g \
     --request-gpus 0 \
     --request-network \
     :pretrain_openwebtext_wordlength \
     :src \
     "export PYTHONPATH=.; ${CMD}"

# run on checkpoints of our model
for STEP in 10000 20000 30000 40000 50000
do
RUN_NAME="eval_openwebtext_wordlength"
EVAL_DATA_DIR=src/wordlength_eval_data
SEED=1111 # random seed
MODEL_DIR=pretrain_openwebtext_wordlength/output/openwebtext_wordlength_SEED${SEED}/checkpoint-${STEP}
CMD="python src/evaluate_wordlength_model.py --model_dir ${MODEL_DIR} --eval_data_dir ${EVAL_DATA_DIR}"
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 32g \
     --request-gpus 0 \
     --request-network \
     :pretrain_openwebtext_wordlength \
     :src \
     "export PYTHONPATH=.; ${CMD}"
 done
