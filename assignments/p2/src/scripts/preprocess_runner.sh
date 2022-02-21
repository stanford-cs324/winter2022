#!/bin/bash

DATA_NAME=wordlength # TODO fill in your own name for your dataset
DATA_DIR=raw_data
OUTPUT_DIR=openwebtext_${DATA_NAME}
mkdir -p $OUTPUT_DIR
NUM_TOTAL_CHUNKS=64
NUM_CHUNKS=10 # only process a subset of the total chunks

for ((CHUNK_IDX=0; CHUNK_IDX < $NUM_CHUNKS; CHUNK_IDX++)); do
    bash src/scripts/process_data.sh $DATA_DIR $OUTPUT_DIR $DATA_NAME $NUM_TOTAL_CHUNKS $CHUNK_IDX &
done

wait

# we tokenize the data in chunks without merging

DATA_DIR=$OUTPUT_DIR
OUTPUT_DIR="."
mkdir -p $OUTPUT_DIR
CACHE=./scr/cache
mkdir -p $CACHE
MODEL_NAME=$1

python src/tokenize_data.py \
    --data_dir $DATA_DIR \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE} \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATA_NAME}
