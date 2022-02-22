#!/bin/bash

set -x

input_dataset_name=openwebtext
output_dataset_name=openwebtext_wordlength
tokenizer_name=gpt2
run_name=preprocess_$output_dataset_name

# Run `./preprocess.sh` on CodaLab.
# Feel free to modify this script.
cl run \
    --name $run_name \
    --request-docker-image sangxie513/cs324-p2-codalab-gpu \
    --request-memory 128g \
    --request-cpus 10 \
    :src \
    :openwebtext \
    "bash src/preprocess.sh $input_dataset_name $output_dataset_name $tokenizer_name"
