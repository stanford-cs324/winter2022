#!/bin/bash

set -x

RUN_NAME="pretrain_openwebtext_wordlength"
DATA_NAME=wordlength
SEED=1111 # random seed
MODEL_NAME=gpt2
WORKSHEET_NAME=cs324-project2 # TODO change thia!!

# TODO FILL THESE HYPERPARAMETERS IN.
# DO NOT EXCEED PER_DEVICE_BATCH_SIZE * ACCUM_STEPS * MAX_STEPS > 400000
PER_DEVICE_BATCH_SIZE=
ACCUM_STEPS=
MAX_STEPS=
LR=
WARMUP_STEPS=
# ============================



CMD="bash src/scripts/pretrain.sh ${DATA_NAME} ${SEED} ${MODEL_NAME} ${PER_DEVICE_BATCH_SIZE} ${ACCUM_STEPS} ${MAX_STEPS} ${LR} ${WARMUP_STEPS}"
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 80g \
     --request-gpus 1 \
     openwebtext_wordlength_tokenized_grouped:preprocess_openwebtext_wordlength/openwebtext_wordlength_tokenized_grouped \
     :src \
     "export PYTHONPATH=.; ${CMD}"
