#!/bin/bash

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="caa"
FILTER_CLASS=1
DEVICE="cpu"
MODEL_TARGET=("tabtransformer" "deepfm" "torchrln" "vime")

model_taget_name=""

for model in "${MODEL_TARGET[@]}"; do
    if [ -n "$model_taget_name" ]; then
        model_taget_name="${model_taget_name}:${model}:${model}"  # Add semicolon separator
    else
        model_taget_name="${model}:${model}"
    fi
done

for DATASET in url ctu_13_neris lcld_v2_iid malware
do
    for MODEL_SRC in tabtransformer deepfm torchrln vime
    do
        list_target_path=""
        for model in "${MODEL_TARGET[@]}"; do
            if [ -n "$list_target_path" ]; then
                list_target_path="${list_target_path}:../models/mlc/best_models/${model}_${DATASET}_default.model:../models/mlc/best_models/${model}_${DATASET}_madry.model"
            else
                list_target_path="../models/mlc/best_models/${model}_${DATASET}_default.model:../models/mlc/best_models/${model}_${DATASET}_madry.model"
            fi
        done
        
        MODEL_PATH_SRC="../models/mlc/best_models/${MODEL_SRC}_${DATASET}_default.model"
        echo CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioC1.py --dataset_name $DATASET --model_name_source $MODEL_SRC --custom_path_source $MODEL_PATH_SRC --model_name_target $model_taget_name --custom_path_target $list_target_path --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --constraints --filter_class $FILTER_CLASS
    done
done

