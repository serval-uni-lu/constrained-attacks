#!/bin/bash

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="moeva"
FILTER_CLASS=1
DEVICE="cpu"


list_n_gen=("100" "50" "200" "200" "100")
list_n_offsprings=("100" "50" "200" "100" "200")

# Ensure both lists have the same length
if [ ${#list_n_gen[@]} -ne ${#list_n_offsprings[@]} ]; then
    echo "Error: The lists have different lengths."
    exit 1
fi

for DATASET in lcld_v2_iid url ctu_13_neris
do
    for MODEL in tabtransformer torchrln deepfm vime
    do
        for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
        do
          for SCENARIO in "--constraints_access --project_name scenario_B1v10" "--no-constraints_access --project_name scenario_B2v10"
          do
            for index in "${!list_n_gen[@]}" 
            do
              n_gen="${list_n_gen[$index]}"
              n_offsprings="${list_n_offsprings[$index]}"
              echo CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE ${SCENARIO} --n_gen $n_gen --n_offsprings $n_offsprings
            done
          done
        done
    done
done



