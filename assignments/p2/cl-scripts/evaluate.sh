#!/bin/bash

set -x

WORKSHEET_NAME=cs324-project2 # TODO change this!!

# first, run on the vanilla gpt2
RUN_NAME="eval_openwebtext_wordlength_step0"
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 32g \
     --request-gpus 0 \
     --request-network \
     :src \
     "bash src/evaluate.sh gpt2 src/wordlength_eval_data"

# run on checkpoints of our model
for STEP in 10000 20000 30000 40000 50000
do
RUN_NAME="eval_openwebtext_wordlength_step${STEP}"
EVAL_DATA_DIR=src/wordlength_eval_data
SEED=1111 # random seed
MODEL_DIR=train1/openwebtext_wordlength_SEED${SEED}/checkpoint-${STEP}
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 32g \
     --request-gpus 0 \
     --request-network \
     :train1 \
     :src \
     "bash src/evaluate.sh ${MODEL_DIR} ${EVAL_DATA_DIR}"
 done
