#!/bin/bash
set -x
CACHE=./scr/cache
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=100000000000
export TORCH_EXTENSIONS_DIR=$CACHE
export WANDB_DISABLED=true

DATA_NAME=${DATA_NAME:-wordlength}
SEED=${SEED:-1111}
MODEL_NAME=${MODEL_NAME:-gpt2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-""}
ACCUM_STEPS=${ACCUM_STEPS:-""}
MAX_STEPS=${MAX_STEPS:-""}
LR=${LR:-""}
WARMUP_STEPS=${WARMUP_STEPS:-""}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

TOKENIZED_DATA=openwebtext_${DATA_NAME}_tokenized_grouped
OUTPUT_DIR=./output
mkdir -p $OUTPUT_DIR

python src/run_clm.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenized_data_dir ${TOKENIZED_DATA} \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --learning_rate $LR \
    --max_steps $MAX_STEPS \
    --output_dir ${OUTPUT_DIR}/openwebtext_${DATA_NAME}_SEED${SEED} \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --save_steps 10000 \
    --seed $SEED \
    --fp16 \
    --warmup_steps ${WARMUP_STEPS} \
    --lr_scheduler_type linear \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUM_STEPS} \
    --max_eval_samples 100 \
    --preprocessing_num_workers 8
# --from_scratch # if you need to train from scratch, use this option

