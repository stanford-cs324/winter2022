#!/bin/bash
CACHE=./cache
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=100000000000
export TORCH_EXTENSIONS_DIR=$CACHE
export WANDB_DISABLED=true

if [[ $# -eq 0 ]] ; then
    echo 'Arguments:'
    echo ''
    echo '--max_steps: Total number of training steps to run.'
    echo '--learning_rate: Highest learning rate in the learning rate schedule (by default, the learning rate schedule increases and decreases linearly, with the peak being this set learning rate)'
    echo '--warmup_steps: Number of steps to do a linear “warmup” from a small learning rate to the desired learning rate'
    echo '--save_steps: how many steps of training before saving (and evaluating on the val set)'

    echo '--per_device_train_batch_size: batch size per GPU. The total batch size is [number of GPUs] * per_device_train_batch_size'
    echo '--gradient_accumulation_steps: Allows for accumulating gradients across multiple sequential steps. This allows you to increase the batch size while trading off computation time. If this parameter is > 1, then the total batch size is [num GPUs] * per_device_train_batch_size * gradient_accumulation_steps'
    echo '--lr_scheduler_type: learning rate scheduler. The default is “linear” which does a linear increase and decrease to and from the learning_rate.'
    echo '--adafactor: use this flag if you want to use the Adafactor optimizer instead of AdamW (default). Adafactor can save on GPU memory by only saving 2 copies of the model instead of 3 needed for AdamW (mean, variance, gradient)'
    echo ''
    echo 'Other possible arguments are listed here: https://github.com/huggingface/transformers/blob/v4.12.5/src/transformers/training_args.py'
    exit 0
fi

data_name=${1:-wordlength}
seed=${2:-1111}
shift 2
rest_args="$@"

TOKENIZED_DATA=openwebtext_${data_name}_tokenized_grouped
OUTPUT_DIR=./output
mkdir -p $OUTPUT_DIR

set -x

python src/run_clm.py \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --fp16 \
    --max_eval_samples 100 \
    --preprocessing_num_workers 8 \
    --tokenized_data_dir ${TOKENIZED_DATA} \
    --output_dir ${OUTPUT_DIR}/openwebtext_${data_name}_SEED${seed} \
    --seed $seed \
    ${rest_args}
# --from_scratch # if you need to train from scratch, use this option

