#!/bin/bash

set -x

RUN_NAME="train1"
DATA_NAME=openwebtext_wordlength
SEED=1111 # random seed
WORKSHEET_NAME=cs324-project2 # TODO change this!!

model_name_or_path=gpt2
# TODO FILL THESE HYPERPARAMETERS IN.
# DO NOT EXCEED ACCUM_STEPS * MAX_STEPS > 100000
per_device_batch_size=
gradient_accumulation_steps=
max_steps=
learning_rate=
warmup_steps=
# ============================

CMD="bash src/train.sh ${DATA_NAME} ${SEED} --model_name_or_path ${model_name_or_path} --per_device_batch_size ${per_device_batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} --max_steps ${max_steps} --learning_rate ${learning_rate} --warmup_steps ${warmup_steps} --save_steps 10000 --lr_scheduler_type linear"
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 80g \
     --request-gpus 1 \
     ${DATA_NAME}:preprocess1/${DATA_NAME} \
     :src \
     "${CMD}"
