#!/bin/bash

DEVICES=0
BATCH_SIZE=1024
ATTACK="caa4"
FILTER_CLASS=1
DEVICE="cpu"
SEED="${SEED:-0}"

MODEL_TARGET=("tabtransformer" "torchrln" "vime" "stg" "tabnet")

for model in "${MODEL_TARGET[@]}"; do
    if [ -n "$model_taget_name" ]; then
        model_taget_name="${model_taget_name}:${model}:${model}:${model}"  # Add semicolon separator
    else
        model_taget_name="${model}:${model}:${model}"
    fi
done

for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_subset.model"
do
    # for SCENARIO in "--no-constraints_access" "--constraints_access"
    for SCENARIO in "--constraints_access"
    do

        if [[ "$DATASET" == "malware" ]]; then
            MAX_EPS=5.0
            SUBSET=100
        else
            MAX_EPS=0.5
            SUBSET=1000
        fi

        list_target_path=""
        for model in "${MODEL_TARGET[@]}"; do
            if [ -n "$list_target_path" ]; then
                list_target_path="${list_target_path}:../models/mlc/best_models/${model}_${DATASET}_default.model:../models/mlc/best_models/${model}_${DATASET}_madry.model:../models/mlc/best_models/${model}_${DATASET}_subset.model"
            else
                list_target_path="../models/mlc/best_models/${model}_${DATASET}_default.model:../models/mlc/best_models/${model}_${DATASET}_madry.model:../models/mlc/best_models/${model}_${DATASET}_subset.model"
            fi
        done
        CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE ${SCENARIO} --filter_class=$FILTER_CLASS --model_name_target $model_taget_name --custom_path_target $list_target_path --seed=$SEED --project_name scenario_D_${DATASET} --load_adv 1 --save_examples 0 --save_adv 0
    done
done
