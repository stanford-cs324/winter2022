#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Main entry point for preprocessing data."
    echo "Usage:"
    echo
    echo "    $0 <input_dataset_name (e.g., openwebtext)> <output_dataset_name (e.g., openwebtext_wordlength)> [additional arguments]"
    echo
    echo "Additional arguments:"
    echo "    --TODO"
    exit 1
fi

CACHE=./cache
mkdir -p $CACHE

TOKENIZER_NAME=$1

input_dataset_name=$1; shift
output_dataset_name=$1; shift
rest_args="$@"

#DATA_NAME=wordlength # TODO fill in your own name for your dataset
DATA_DIR=openwebtext
#OUTPUT_DIR=openwebtext_${DATA_NAME}
mkdir -p $OUTPUT_DIR
NUM_TOTAL_CHUNKS=64
NUM_CHUNKS=10 # only process a subset of the total chunks

# Do chunks in parallel
for ((CHUNK_IDX=0; CHUNK_IDX < $NUM_CHUNKS; CHUNK_IDX++)); do
    python src/process_data.py \
        --data_dir $input_dataset_name \
        --output_dir $output_dataset_name \
        --dataset_name $DATA_NAME \
        --total_chunks $NUM_TOTAL_CHUNKS \
        --chunk_idx $CHUNK_IDX &
done

wait

# we tokenize the data in chunks without merging

DATA_DIR=$OUTPUT_DIR
OUTPUT_DIR="."
mkdir -p $OUTPUT_DIR
CACHE=./scr/cache
mkdir -p $CACHE

python src/tokenize_data.py \
    --data_dir $DATA_DIR \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE} \
    --model_name ${TOKENIZER_NAME} \
    --dataset_name ${DATA_NAME}
