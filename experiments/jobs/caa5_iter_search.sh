#!/bin/bash

DEVICES=-1
BATCH_SIZE=1024
ATTACK="caa5"
FILTER_CLASS=1
DEVICE="cpu"
SEED="${SEED:-0}"

n_gen=100
n_offsprings=100
STEPS=10
MAX_EPS=0.5

# for DATASET in url lcld_v2_iid wids ctu_13_neris 
for DATASET in ctu_13_neris 
do
    for SEED in 0 1 2 3 4
    do
        # for DATASET in lcld_v2_iid url ctu_13_neris malware wids
        # for n_gen in 50 200 1000
        for n_gen in 1000
        do
            if [[ "$DATASET" == "malware" ]]; then
                SUBSET=100
            else
                SUBSET=1000
            fi
            # for MODEL in vime tabnet stg torchrln tabtransformer
            for MODEL in tabnet
            do
                for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
                do
                    for SCENARIO in "--constraints_access"
                    do
                        # echo sbatch ./experiments/jobs/launch-cpu.sh
                        sbatch ./experiments/jobs/launch-cpu.sh "CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings" --project_name caa5_iter_search_v3_${DATASET} --save_adv 1 --steps=$STEPS # scenario_AB_${DATASET}
                    done
                done
            done
        done
    done
done
