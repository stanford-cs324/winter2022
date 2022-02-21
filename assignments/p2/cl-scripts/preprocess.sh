#!/bin/bash

WORKSHEET_NAME=cs324-project2 # TODO change thia!!

RUN_NAME="preprocess1"
TOKENIZER_NAME=gpt2
INPUT_DATA_DIR=openwebtext
OUTPUT_DATA_DIR=openwebtext_wordlength
CMD="bash src/preprocess.sh ${INPUT_DATA_DIR} ${OUTPUT_DATA_DIR} ${TOKENIZER_NAME}"
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 128g \
     --request-gpus 0 \
     --request-cpus 10 \
     --request-network \
     :openwebtext \
     :src \
     "${CMD}"
