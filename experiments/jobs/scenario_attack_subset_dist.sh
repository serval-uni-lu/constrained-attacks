#!/bin/sh

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="caa3"
FILTER_CLASS=1
DEVICE="cpu"

n_gen=100
n_offsprings=100

for SEED in 0 1 2 3 4
do 
    for DATASET in ctu_13_neris
    do
        for MODEL in tabtransformer torchrln vime stg tabnet
        do
            for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_dist.model" "../models/mlc/best_models/${MODEL}_${DATASET}_subset.model"
            do
                for SCENARIO in "--constraints_access" "--no-constraints_access"
                do
                    sbatch ./experiments/jobs/launch-cpu.sh "CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings" --project_name scenario_attack_${DATASET}_v1 --save_adv 1
                done
            done
        done
    done
done

# expected 72 * 5 per xp
