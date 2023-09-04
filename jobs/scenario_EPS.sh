#!/bin/sh

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="pgdl2+apgd"
FILTER_CLASS=1
DEVICE="cpu"


A1="--constraints_access False --project_name scenario_EPS"
A2="--constraints_access True --project_name scenario_EPS"

for DATASET in lcld_v2_iid
do
    for MODEL in torchrln
    do
        for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
        do
            for SCENARIO in $A1 $A2
            do
                for MAX_EPS in 0.05 0.1 0.2 0.5 1.0 2.0 5.0
                do
                    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO}
                done
            done
        done
    done
done

