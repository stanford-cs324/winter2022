#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Main entry point for preprocessing data. Make sure the output directories don't exist already when running this script (e.g., from a previous failed run)"
    echo "Usage:"
    echo
    echo "    $0 <DATA_DIR (e.g., openwebtext)> <OUTPUT_DIR (e.g., openwebtext_wordlength)> <TOKENIZER_NAME (e.g., gpt2)> <NUM_TOTAL_CHUNKS (e.g., 64)> <NUM_CHUNKS (e.g., 10)>"
    echo
    exit 1
fi

CACHE=./cache
mkdir -p $CACHE

DATA_DIR=$1
OUTPUT_DIR=$2
TOKENIZER_NAME=$3

mkdir -p $OUTPUT_DIR
NUM_TOTAL_CHUNKS=${4:-64}
NUM_CHUNKS=${5:-10} # only process a subset of the total chunks

for ((CHUNK_IDX=0; CHUNK_IDX < $NUM_CHUNKS; CHUNK_IDX++)); do
    python src/process_data.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --total_chunks $NUM_TOTAL_CHUNKS \
        --chunk_idx $CHUNK_IDX &
done

wait

# we tokenize the data in chunks without merging

DATA_DIR=$OUTPUT_DIR
CACHE=./cache
mkdir -p $CACHE

python src/tokenize_data.py \
    --data_dir $DATA_DIR \
    --cache_dir ${CACHE} \
    --model_name ${TOKENIZER_NAME} \
