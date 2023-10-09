#!/bin/bash

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
# MAX_EPS=0.5
ATTACK="moeva"
FILTER_CLASS=1
DEVICE="cpu"
SEED="${SEED:-0}"

n_gen=100
n_offsprings=100

for DATASET in lcld_v2_iid url ctu_13_neris
do
    for MODEL in tabtransformer torchrln vime
    do
        for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
        do
          for SCENARIO in "--constraints_access --project_name scenario_B1v11_EPS" "--no-constraints_access --project_name scenario_B2v11_EPS"
          do
            for MAX_EPS in 0.25 0.5 1
            do
              sbatch ./jobs/launch-cpu.sh "CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE ${SCENARIO} --filter_class=$FILTER_CLASS --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings"
            done
          done
        done
    done
done



