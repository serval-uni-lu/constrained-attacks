#!/bin/sh

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="pgdl2+apgd+caa3"
FILTER_CLASS=1
DEVICE="cpu"
SEED="${SEED:-0}"

SCENARIO="--constraints_access --project_name scenario_A1v18"

for DATASET in lcld_v2_iid url ctu_13_neris
do
    for MODEL in tabtransformer torchrln vime 
    do
        for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
        do
            echo CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --seed=$SEED
        done
    done
done

