#!/bin/bash

DEVICES=0
SUBSET=100
BATCH_SIZE=1024
MAX_EPS=5.0
ATTACK="caa4"
FILTER_CLASS=1
DEVICE="cpu"

MODEL_TARGET=("tabtransformer" "torchrln" "vime" "stg" "tabnet")
model_taget_name=""
for model in "${MODEL_TARGET[@]}"; do
    if [ -n "$model_taget_name" ]; then
        model_taget_name="${model_taget_name}:${model}:${model}"  # Add semicolon separator
    else
        model_taget_name="${model}:${model}"
    fi
done

for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" #"../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
do
    for SCENARIO in "--constraints_access"
    do
        list_target_path=""
        for model in "${MODEL_TARGET[@]}"; do
            if [ -n "$list_target_path" ]; then
                list_target_path="${list_target_path}:../models/mlc/best_models/${model}_${DATASET}_default.model:../models/mlc/best_models/${model}_${DATASET}_madry.model"
            else
                list_target_path="../models/mlc/best_models/${model}_${DATASET}_default.model:../models/mlc/best_models/${model}_${DATASET}_madry.model"
            fi
        done
        CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE ${SCENARIO} --filter_class=$FILTER_CLASS --model_name_target $model_taget_name --custom_path_target $list_target_path --seed=$SEED --project_name transferability_${DATASET}_v1 --load_adv 1 --save_adv 0 --save_examples 0
    done
done
