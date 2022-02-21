#!/bin/bash

WORKSHEET_NAME=cs324-project2 # TODO change thia!!

RUN_NAME="preprocess_openwebtext_wordlength"
TOKENIZER_NAME=gpt2
CMD="bash src/scripts/preprocess_runner.sh ${TOKENIZER_NAME}"
cl run \
     -n ${RUN_NAME} \
     -w ${WORKSHEET_NAME} \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 128g \
     --request-gpus 0 \
     --request-cpus 8 \
     --request-network \
     :openwebtext \
     :src \
     "export PYTHONPATH=.; ${CMD}"
