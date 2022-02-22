#!/bin/bash

# Run `./evaluate.sh` on CodaLab.
# Feel free to modify this script.

set -x

dataset_name=openwebtext_wordlength
eval_data=wordlength_eval_data
seed=1111

# Evaluate the GPT-2 model
run_name="eval_${dataset_name}_step0"
cl run \
     --name $run_name \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 32g \
     :src \
     :$eval_data \
     "bash src/evaluate.sh gpt2 wordlength_eval_data"

# Evaluate on checkpoints of the continued-pretraining model.
for step in 10000 20000 30000 40000 50000; do
  run_name=eval_${dataset_name}_step${step}
  train_dir=train_${dataset_name}_seed${seed}
  model_dir=${train_dir}/checkpoint-${step}
  eval_data_dir=src/wordlength_eval_data
  cl run \
     --name $run_name \
     --request-docker-image sangxie513/cs324-p2-codalab-gpu \
     --request-memory 32g \
     :src \
     :$eval_data \
     .:$train_dir \
     "bash src/evaluate.sh ${model_dir} ${eval_data}"
done
