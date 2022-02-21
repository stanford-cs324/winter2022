#!/bin/bash

TOKENIZER_NAME=$1

DATA_NAME=wordlength # TODO fill in your own name for your dataset
DATA_DIR=openwebtext
OUTPUT_DIR=openwebtext_${DATA_NAME}
mkdir -p $OUTPUT_DIR
NUM_TOTAL_CHUNKS=64
NUM_CHUNKS=10 # only process a subset of the total chunks

for ((CHUNK_IDX=0; CHUNK_IDX < $NUM_CHUNKS; CHUNK_IDX++)); do
    python src/process_data.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
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
