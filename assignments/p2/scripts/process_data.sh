#!/bin/bash

DATA_DIR=$1
OUTPUT_DIR=$2
DATA_NAME=$3
NUM_CHUNKS=${4-32}
CHUNK_IDX=${5-0}

python src/process_data.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATA_NAME \
    --total_chunks $NUM_CHUNKS \
    --chunk_idx $CHUNK_IDX
